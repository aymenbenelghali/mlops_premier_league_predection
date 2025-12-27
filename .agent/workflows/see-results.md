---
description: How to view recent match results and check application status
---

1. Start the application if it's not already running:
// turbo
```powershell
.\.venv_cuda\Scripts\python.exe app.py
```

2. Check the recent results endpoint directly:
```powershell
curl http://localhost:5000/api/recent_results
```

3. Open the web interface to see the results in the UI:
- Open `http://localhost:5000` in your browser.
- Scroll down to the "Recent Results" section.
- Click "Refresh" if needed.
