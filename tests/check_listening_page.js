// Exercise the generated page in an isolated DOM fixture; no study ratings saved.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync(process.argv[2], 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const all = [];
class Element {
  constructor(tag) { this.tag = tag; this.value = ''; this.dataset = {}; this.events = {}; this.children = []; all.push(this); }
  append(...children) { this.children.push(...children); }
  setAttribute() {}
  addEventListener(name, fn) { this.events[name] = fn; }
  click() {}
  focus() {}
}
const elements = Object.fromEntries(['trials','listener','progress','download'].map(id => [id,new Element(id)]));
let exported;
const storage = new Map();
const context = {
  document: { getElementById: id => elements[id], createElement: tag => new Element(tag),
    querySelectorAll: () => all.filter(el => el.dataset.key) },
  localStorage: { getItem: key => storage.get(key), setItem: (key,value) => storage.set(key,value) },
  Blob, URL: { createObjectURL: blob => { exported = blob; return 'fixture:csv'; }, revokeObjectURL() {} },
  setTimeout() {}
};
vm.runInNewContext(source, context);
assert.equal(elements.trials.children.length,6);
assert.equal(all.filter(el => el.tag === 'audio').length,30);
assert.equal(all.filter(el => el.tag === 'select').length,96);
assert.match(elements.progress.textContent,/0 \/ 24/);
elements.listener.value = 'UI_TEST_FIXTURE';
elements.download.events.click();
(async () => {
  const csv = await exported.text();
  assert.equal(csv.trim().split('\n').length,25,'CSV must contain real newlines');
  assert.ok(csv.startsWith('"listener_id","trial_id","sample_code"'));
  for (const select of all.filter(el => el.tag === 'select')) select.value = '3';
  elements.listener.events.input();
  assert.match(elements.progress.textContent,/24 \/ 24/);
  elements.download.events.click();
  const completed = await exported.text();
  assert.equal(completed.trim().split('\n').length,25);
  assert.ok(completed.includes('"3","3","3","3"'));
  elements.listener.value = 'SECOND_UI_FIXTURE';
  elements.listener.events.input();
  assert.match(elements.progress.textContent,/0 \/ 24/,'New listener must not inherit ratings');
  elements.listener.value = 'UI_TEST_FIXTURE';
  elements.listener.events.input();
  assert.match(elements.progress.textContent,/24 \/ 24/,'Returning listener must resume their own session');
  console.log('PASS: six trials, 30 audio controls, 96 fields, separate listener sessions and CSV export');
})().catch(error => { console.error(error); process.exitCode=1; });
