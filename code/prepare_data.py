#!/usr/bin/env python3
"""Fetch only frozen-manifest WAVs from the official 48 kHz ZIP archives.

Python standard library + curl; HTTP ranges avoid downloading ~5 GB of unused
audio. ZIP CRCs are checked for downloaded AND existing files. No WAV is changed.
Alternatively pass --archive-dir with the four downloaded ZIPs and logfiles.zip.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
BITSTREAMS = {
    'logfiles': '11185dc8-9cf1-405b-b858-35bd6a04aedd',
    'clean_trainset_28spk_wav': '245452b6-6235-44b6-a6f9-e7eb19797769',
    'noisy_trainset_28spk_wav': 'ecb5a102-bb00-46d3-8af5-40c79823b837',
    'clean_testset_wav': 'dec213d3-bf57-4777-9663-c24bdce92d5e',
    'noisy_testset_wav': '13c1bfbf-14a6-41db-9b41-8f7310f01ad5',
}


def read_manifests(root):
    """Reject changed scope, duplicates, overlaps, and invalid file names."""
    manifests = {}
    for split, count in [('tuning', 400), ('evaluation', 240), ('exploratory', 3)]:
        path = root / 'data' / f'{split}_manifest.csv'
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ['File', 'Noise', 'NominalSNR']:
                raise ValueError(f'{path}: expected File,Noise,NominalSNR columns')
            rows = list(reader)
        names = [r['File'] for r in rows]
        if len(rows) != count or len(set(names)) != count:
            raise ValueError(f'{path}: expected {count} distinct rows')
        for row in rows:
            if not re.fullmatch(r'p\d{3}_\d{3}', row['File']):
                raise ValueError(f'{path}: invalid file name {row["File"]!r}')
            row['NominalSNR'] = float(row['NominalSNR'])
        manifests[split] = rows
    for a, b in [('tuning', 'evaluation'), ('tuning', 'exploratory'),
                 ('evaluation', 'exploratory')]:
        if {r['File'] for r in manifests[a]} & {r['File'] for r in manifests[b]}:
            raise ValueError(f'{a}/{b}: overlapping files')
    train_noises = 'babble cafeteria car kitchen meeting metro restaurant station traffic ssn'.split()
    expected_train = Counter({(n, s): 10 for n in train_noises for s in [0, 5, 10, 15]})
    expected_test = Counter({(n, s, p): 10 for n in ['bus', 'office', 'cafe']
                             for s in [2.5, 7.5, 12.5, 17.5] for p in ['p232', 'p257']})
    if Counter((r['Noise'], r['NominalSNR']) for r in manifests['tuning']) != expected_train:
        raise ValueError('Tuning noise/SNR balance differs from the frozen design')
    if Counter((r['Noise'], r['NominalSNR'], r['File'].split('_')[0])
               for r in manifests['evaluation']) != expected_test:
        raise ValueError('Evaluation noise/SNR/speaker balance differs from the frozen design')
    expected_examples = {('p257_004', 'bus', 2.5), ('p257_017', 'office', 2.5),
                         ('p257_009', 'cafe', 2.5)}
    if {(r['File'], r['Noise'], r['NominalSNR']) for r in manifests['exploratory']} != expected_examples:
        raise ValueError('Exploratory manifest differs from the frozen design')
    return manifests


def fetch(url, byte_range=None):
    """Use curl's system certificate trust; never disable TLS verification."""
    with tempfile.TemporaryDirectory(prefix='eecs351-http-') as tmp:
        body, headers = Path(tmp) / 'body', Path(tmp) / 'headers'
        command = ['curl', '--fail', '--location', '--silent', '--show-error',
                   '--retry', '4', '--retry-delay', '2', '--connect-timeout', '20',
                   '--max-time', '180', '--dump-header', str(headers), '--output', str(body)]
        if byte_range is not None:
            limit = int(byte_range[1:]) if byte_range.startswith('-') else \
                int(byte_range.split('-')[1])-int(byte_range.split('-')[0])+1
            command += ['--range', byte_range, '--max-filesize', str(limit)]
        subprocess.run(command + [url], check=True)
        response_headers = headers.read_text()
        content = body.read_bytes()
        if byte_range is not None:
            ranges = re.findall(r'^Content-Range:\s*bytes (\d+)-(\d+)/(\d+)',
                                response_headers, flags=re.I | re.M)
            if not ranges:
                raise ValueError(f'{url}: HTTP ranges unavailable; use --archive-dir')
            start, end, total = map(int, ranges[-1])
            expected = (max(0, total-int(byte_range[1:])), total-1) if byte_range.startswith('-') \
                else tuple(map(int, byte_range.split('-')))
            if (start, end) != expected or len(content) != end-start+1:
                raise ValueError(f'{url}: invalid or truncated range response')
            return content, total
        return content, len(content)


class RangeReader(io.RawIOBase):
    """Seekable read-only HTTP stream for zipfile's central-directory parser."""
    def __init__(self, url):
        self.url = url
        self.tail, self.size = fetch(url, '-65557')
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position+size)
        if end <= self.position:
            return b''
        if self.position >= self.size-len(self.tail):
            result = self.tail[self.position-(self.size-len(self.tail)):end-(self.size-len(self.tail))]
        else:
            result, total = fetch(self.url, f'{self.position}-{end-1}')
            if total != self.size:
                raise ValueError('Archive size changed during download')
        self.position = end
        return result


def remote_member(url, info):
    """Extract one entry using central-directory sizes, including ZIP64 sizes."""
    offset = info.header_offset
    header, _ = fetch(url, f'{offset}-{offset+29}')
    values = struct.unpack('<4s5H3I2H', header)
    if values[0] != b'PK\x03\x04' or values[2] & 1 or values[3] != info.compress_type:
        raise ValueError(f'{info.filename}: invalid/encrypted ZIP member')
    name_len, extra_len = values[-2:]
    length = name_len+extra_len+info.compress_size
    payload, _ = fetch(url, f'{offset+30}-{offset+29+length}')
    encoding = 'utf-8' if values[2] & 0x800 else 'cp437'
    if payload[:name_len].decode(encoding) != info.filename:
        raise ValueError('Local and central ZIP file names disagree')
    compressed = payload[name_len+extra_len:]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(compressed, -15)
    if info.compress_type == zipfile.ZIP_STORED:
        return compressed
    raise ValueError(f'Unsupported ZIP compression: {info.compress_type}')


def check_crc(content, info):
    return len(content) == info.file_size and zlib.crc32(content) & 0xffffffff == info.CRC


def validate_logs(manifests, data_root):
    for split, rows in manifests.items():
        name = 'log_trainset_28spk.txt' if split == 'tuning' else 'log_testset.txt'
        conditions = {}
        for line in (data_root / name).read_text().splitlines():
            if not line.strip():
                continue
            file, noise, snr = line.split()
            if file in conditions:
                raise ValueError(f'{name}: duplicate log entry {file}')
            conditions[file] = (noise, float(snr))
        for row in rows:
            if conditions.get(row['File']) != (row['Noise'], row['NominalSNR']):
                raise ValueError(f'{split}: official log mismatch for {row["File"]}')


def _prepare(root, data_root, archive_dir, workers):
    manifests = read_manifests(root)
    print('Frozen manifests: 400 tuning + 240 evaluation + 3 exploratory; balance and exclusions PASS', flush=True)
    data_root.mkdir(parents=True, exist_ok=True)
    log_path = archive_dir / 'logfiles.zip' if archive_dir else None
    log_url = f'https://datashare.ed.ac.uk/server/api/core/bitstreams/{BITSTREAMS["logfiles"]}/content'
    with zipfile.ZipFile(log_path or io.BytesIO(fetch(log_url)[0])) as archive:
        for name in ['log_readme.txt', 'log_trainset_28spk.txt', 'log_testset.txt']:
            content = archive.read(name)  # zipfile verifies the CRC
            target = data_root / name
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'{target}: existing log differs; inspect before replacing')
            target.write_bytes(content)
    validate_logs(manifests, data_root)
    print('Official condition logs: all 643 manifest labels/SNRs PASS', flush=True)
    inventory = []
    for archive_name, bitstream in BITSTREAMS.items():
        if archive_name == 'logfiles':
            continue
        rows = manifests['tuning'] if 'trainset' in archive_name else manifests['evaluation'] + manifests['exploratory']
        url = f'https://datashare.ed.ac.uk/server/api/core/bitstreams/{bitstream}/content'
        local = archive_dir / (archive_name+'.zip') if archive_dir else None
        source = local if local else RangeReader(url)
        with zipfile.ZipFile(source) as archive:
            index = {}
            for info in archive.infolist():
                if info.is_dir() or info.filename.startswith('__MACOSX/'):
                    continue
                name = Path(info.filename).name
                if name in index:
                    raise ValueError(f'{archive_name}: ambiguous archive basename {name}')
                index[name] = info
            selected = [index[r['File']+'.wav'] for r in rows]

            def extract(info):
                target = data_root / archive_name / Path(info.filename).name
                content = target.read_bytes() if target.exists() else None
                if content is not None and not check_crc(content, info):
                    raise ValueError(f'{target}: existing WAV fails official ZIP CRC; inspect/remove and rerun')
                if content is None:
                    if local:
                        with zipfile.ZipFile(local) as local_zip:
                            content = local_zip.read(info.filename)
                    else:
                        content = remote_member(url, info)
                    if not check_crc(content, info):
                        raise ValueError(f'{target}: downloaded WAV fails ZIP size/CRC validation')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix('.wav.part')
                    temporary.write_bytes(content)
                    temporary.replace(target)
                return {'path': str(target.relative_to(data_root)), 'bytes': len(content),
                        'crc32': f'{info.CRC:08x}', 'sha256': hashlib.sha256(content).hexdigest(),
                        'source': url}

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(extract, info) for info in selected]
                for i, future in enumerate(as_completed(futures), 1):
                    inventory.append(future.result())
                    if i % 50 == 0 or i == len(selected):
                        print(f'{archive_name}: {i}/{len(selected)} CRC-verified', flush=True)
    output = root / 'outputs' / 'setup'
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'data_root': str(data_root), 'pair_count': 643, 'wav_count': len(inventory),
              'manifest_sha256': {s: hashlib.sha256((root/'data'/f'{s}_manifest.csv').read_bytes()).hexdigest()
                                  for s in manifests},
              'files': sorted(inventory, key=lambda x: x['path'])}
    (output/'download_inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'PASS: {len(inventory)} original WAVs ready. Inventory: {output / "download_inventory.json"}')


def prepare(root, data_root, archive_dir, workers):
    # Never leave a previous PASS report in place after a failed rerun.
    output = root / 'outputs' / 'setup'
    output.mkdir(parents=True, exist_ok=True)
    path = output/'download_inventory.json'
    state = {'status': 'IN_PROGRESS', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'data_root': str(data_root)}
    path.write_text(json.dumps(state, indent=2)+'\n')
    try:
        _prepare(root, data_root, archive_dir, workers)
    except Exception as error:
        state.update(status='FAIL', error=str(error))
        path.write_text(json.dumps(state, indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data')
    parser.add_argument('--archive-dir', type=Path, help='Offline folder containing the five official ZIP archives')
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--check-manifests', action='store_true', help='Validate CSV scope only; no download')
    args = parser.parse_args()
    try:
        if args.check_manifests:
            read_manifests(ROOT)
            print('PASS: frozen manifests, condition balance, speaker balance, and exclusions')
        else:
            prepare(ROOT, args.data_root.resolve(), args.archive_dir.resolve() if args.archive_dir else None, args.workers)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f'DATA SETUP FAILED: {error}\n')
