# 🧠 ML Portfolio: Fastai, Apps & More

Welcome! This repository is a growing collection of my machine learning projects, blog posts, and web apps.

---

## 📚 Blog (Quarto)

The `blog/` folder contains my learning notes, experiments, and reflections — all written using [Quarto](https://quarto.org/) and deployed as a static website via GitHub Pages.

### 📖 Read the blog here:
👉 [https://AmeNach.github.io/my-portfolio](https://AmeNach.github.io/my-portfolio)

---

## 🧪 Web Apps

Each app lives in its own folder under `apps/`, with isolated dependencies using [`uv`](https://github.com/astral-sh/uv).

| App Name               | Description                             | Live Demo                                      |
|------------------------|-----------------------------------------|------------------------------------------------|
| `CatVsDog-classifier`  | Fastai model for image classification   | [Demo]()                                       |
| Coming soon            |                                         | Coming soon                                    |

---

## 📁 Project Structure

my-portfolio/
├── blog/ ← Quarto blog
├── apps/ ← Web apps (Gradio/Streamlit/etc.)
│ ├── image-classifier/
├── README.md
├── LICENSE
└── .gitignore

## 🛠️ Environment Setup (via `uv`)

From within each app folder:

```bash
uv venv
uv pip install -r requirements.txt
python app.py
```

## 📜 License

This project is open source under the MIT License. Feel free to explore, use, and contribute!

## 🤝 Contact

GitHub: @AmeNach

LinkedIn: Amel Nait Achour