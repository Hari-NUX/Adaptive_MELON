# Local Gmail OAuth files

Keep OAuth credentials and tokens in this folder on your own computer. Git ignores every file here except this guide and `.gitkeep`.

1. In Google Cloud Console, create/select a project and enable the Gmail API.
2. Configure the OAuth consent screen. For a personal test app, use External/testing and add your own Google account as a test user if Google asks.
3. Create an OAuth client ID with application type **Desktop app**, then download its JSON file.
4. Place the downloaded file here as `secrets/credentials.json`.
5. Start the floating app. On first Gmail refresh, sign in through the browser and approve read and draft access. The app stores its refresh token locally as `secrets/gmail_token.json`.

The requested Gmail scopes are `gmail.readonly` and `gmail.compose`. Google's `gmail.compose` scope permits API clients to send as well as manage drafts, but this local app contains draft creation only and has no send operation. Review the recipient and message before saving a draft. Never commit or share `credentials.json` or `gmail_token.json`.

Google's setup guide: https://developers.google.com/workspace/gmail/api/quickstart/python
Gmail scope descriptions: https://developers.google.com/workspace/gmail/api/auth/scopes
