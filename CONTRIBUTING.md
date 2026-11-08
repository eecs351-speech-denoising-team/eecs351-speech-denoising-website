# Working together

We edit directly on `main`. Pull requests are optional.

## First time

```sh
git clone https://github.com/juliusl-code/eecs351-speech-denoising.git
cd eecs351-speech-denoising
```

You need collaborator access to this private repository and your own GitHub authentication.

## Each editing session

1. Start with `git pull --ff-only` before editing.
2. Tell teammates which page you are working on. Avoid editing the same file simultaneously, especially shared navigation and `assets/style.css`.
3. Edit the files yourself or with an agent. Open `index.html` locally to check the pages and links.
4. Review `git diff`, then stage the files you changed and commit:

```sh
git add index.html  # replace with the files you changed
git commit -m "Describe your update"
git pull --rebase
git push origin main
```

If Git reports a conflict, resolve it with your teammate or agent and review the merged content before pushing. Never force-push over teammates' changes. If a push is rejected because someone pushed first, pull with rebase again and retry.

## Content conventions

- Keep the site concise and in English.
- Report plans as plans; do not invent experimental results.
- Home ends at Project report. Technical Details and Results remain title-only until content is ready.
- What We Learned and Our Team are currently hidden from navigation.
- Keep original audio datasets out of Git. Store dataset download instructions instead.
- Do not enable GitHub Pages or change repository visibility until the team decides to publish.
