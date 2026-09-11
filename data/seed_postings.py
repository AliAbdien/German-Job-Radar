"""Hand-written seed training set: 26 realistic German job postings, each with
a manually-verified gold JobExtraction label.

Why hand-written rather than pulled from real scraped postings: redistributing
scraped third-party job-ad text in a public training-data file is a real
copyright/ToS problem (most job boards' terms explicitly forbid it), even
when the *code* that trained on it is open source. These are written from
scratch to be *representative* of real German postings - formal corporate
German, startup Denglisch, missing salary (the norm, not the exception),
implicit seniority signals, mixed remote policies - without being a copy of
any specific real listing. Real, currently-live postings only enter this
project at *evaluation* time, pulled fresh from the official Bundesagentur
für Arbeit Jobsuche API (src/jobsuche_client.py) - never stored or redistributed.

Run `python -m data.seed_postings` to validate every label against the
pydantic schema and write data/seed_postings.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema import JobExtraction  # noqa: E402

OUT_PATH = Path(__file__).parent / "seed_postings.jsonl"

EXAMPLES: list[dict] = [
    {
        "text": (
            "Junior Data Scientist (m/w/d) – München\n\n"
            "Beispiel GmbH sucht zum nächstmöglichen Zeitpunkt einen Junior Data "
            "Scientist. Du analysierst große Datenmengen, baust ML-Modelle und "
            "arbeitest eng mit dem Engineering-Team zusammen.\n\n"
            "Anforderungen:\n- Abgeschlossenes Studium in Informatik, Mathematik "
            "oder vergleichbar\n- Python, SQL, erste Erfahrung mit Machine Learning\n"
            "- Deutsch (C1), Englisch (B2)\n\nWünschenswert: PyTorch, Docker\n\n"
            "Wir bieten: 48.000 - 58.000 € brutto/Jahr, hybrides Arbeiten "
            "(2 Tage Homeoffice), Standort München."
        ),
        "label": {
            "job_title": "Junior Data Scientist",
            "company": "Beispiel GmbH",
            "seniority": "junior",
            "required_skills": ["Python", "SQL", "Machine Learning"],
            "nice_to_have_skills": ["PyTorch", "Docker"],
            "salary": {"min_eur": 48000, "max_eur": 58000, "stated": True},
            "location": "München",
            "remote_policy": "hybrid",
            "required_languages": ["German (C1)", "English (B2)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "(Senior) Machine Learning Engineer – 100% Remote (DE)\n\n"
            "Hey! We're a small AI startup building tools for logistics. Looking "
            "for someone who's shipped production ML systems before and can own "
            "our model serving stack end to end.\n\n"
            "Must-haves: 4+ years building & deploying ML models, strong Python, "
            "experience with PyTorch or TensorFlow, comfortable with AWS/GCP.\n"
            "Nice-to-haves: Kubernetes, MLOps tooling (MLflow, Weights & Biases).\n\n"
            "English is our working language, German not required. Fully remote "
            "within Germany. We sponsor visas for the right candidate."
        ),
        "label": {
            "job_title": "Senior Machine Learning Engineer",
            "company": None,
            "seniority": "senior",
            "required_skills": ["Python", "PyTorch", "TensorFlow", "AWS", "GCP"],
            "nice_to_have_skills": ["Kubernetes", "MLflow", "Weights & Biases"],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": None,
            "remote_policy": "remote",
            "required_languages": ["English"],
            "visa_sponsorship_mentioned": True,
        },
    },
    {
        "text": (
            "Werkstudent (w/m/d) Data Engineering – Berlin\n\n"
            "Zur Unterstützung unseres Data-Teams suchen wir ab sofort einen "
            "Werkstudenten im Bereich Data Engineering (15-20h/Woche).\n\n"
            "Du solltest mitbringen:\n- Studium der Informatik oder verwandter "
            "Fachrichtung\n- Grundkenntnisse in SQL und Python\n- Interesse an "
            "Datenpipelines\n\nDeutschkenntnisse fließend erforderlich, gute "
            "Englischkenntnisse von Vorteil.\n\nBüro in Berlin-Mitte, 2x/Woche "
            "vor Ort gewünscht."
        ),
        "label": {
            "job_title": "Werkstudent Data Engineering",
            "company": None,
            "seniority": "working_student",
            "required_skills": ["SQL", "Python"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Berlin",
            "remote_policy": "hybrid",
            "required_languages": ["German (fluent)", "English"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "AI/ML Engineer – Stuttgart | Automotive\n\n"
            "MAHLE entwickelt KI-Lösungen für die Fertigung der Zukunft. Für "
            "unser Project House suchen wir einen AI/ML Engineer.\n\n"
            "Ihr Profil:\n- Mehrjährige Berufserfahrung in ML/Data Engineering\n"
            "- Sehr gute Kenntnisse in Python, PyTorch oder TensorFlow\n- "
            "Erfahrung mit Zeitreihendaten von Sensoren von Vorteil\n- Deutsch "
            "und Englisch verhandlungssicher\n\nWir bieten ein attraktives "
            "Gehaltspaket, Standort Stuttgart, flexible Arbeitszeiten."
        ),
        "label": {
            "job_title": "AI/ML Engineer",
            "company": "MAHLE",
            "seniority": "mid",
            "required_skills": ["Python", "PyTorch", "TensorFlow"],
            "nice_to_have_skills": ["Time Series Analysis", "Sensor Data"],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Stuttgart",
            "remote_policy": "unspecified",
            "required_languages": ["German (fluent)", "English (fluent)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Praktikum Data Science (6 Monate) – Hamburg\n\n"
            "Für unser Analytics-Team suchen wir ab sofort einen Praktikanten "
            "im Bereich Data Science. Du unterstützt bei der Entwicklung von "
            "Prognosemodellen für unser E-Commerce-Geschäft.\n\nVoraussetzung: "
            "laufendes Studium, Grundkenntnisse Python/R, Interesse an "
            "Statistik. Kein Gehalt angegeben, Vergütung nach Absprache."
        ),
        "label": {
            "job_title": "Praktikum Data Science",
            "company": None,
            "seniority": "intern",
            "required_skills": ["Python", "Statistics"],
            "nice_to_have_skills": ["R"],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Hamburg",
            "remote_policy": "unspecified",
            "required_languages": [],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Lead AI Engineer (all genders) – Frankfurt am Main\n\n"
            "Als Lead AI Engineer übernimmst du die fachliche Führung unseres "
            "5-köpfigen ML-Teams und verantwortest die technische Roadmap "
            "unserer NLP-Produkte.\n\nAnforderungen: 6+ Jahre Erfahrung in ML "
            "Engineering, davon mindestens 2 Jahre in leitender Funktion, "
            "tiefes Verständnis von LLMs und NLP, sehr gute Python-Kenntnisse. "
            "Deutschkenntnisse von Vorteil, aber nicht zwingend.\n\nGehalt: "
            "95.000 - 120.000 € je nach Erfahrung. Vollständig remote möglich, "
            "Büro optional in Frankfurt."
        ),
        "label": {
            "job_title": "Lead AI Engineer",
            "company": None,
            "seniority": "lead",
            "required_skills": ["Python", "NLP", "LLMs", "Team Leadership"],
            "nice_to_have_skills": ["German"],
            "salary": {"min_eur": 95000, "max_eur": 120000, "stated": True},
            "location": "Frankfurt am Main",
            "remote_policy": "remote",
            "required_languages": [],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Computer Vision Engineer (m/w/d) – Nürnberg\n\n"
            "Wir suchen einen Computer Vision Engineer zur Weiterentwicklung "
            "unserer Qualitätskontrolle in der Fertigung. Du arbeitest mit "
            "Anomalieerkennung und Bildverarbeitung.\n\nMust-have: PyTorch oder "
            "TensorFlow, OpenCV, Erfahrung mit Anomaly Detection oder Object "
            "Detection. Deutsch mindestens B2, Englisch B2.\n\nUnbefristete "
            "Festanstellung, Standort Nürnberg, vor Ort erforderlich."
        ),
        "label": {
            "job_title": "Computer Vision Engineer",
            "company": None,
            "seniority": "unspecified",
            "required_skills": ["PyTorch", "TensorFlow", "OpenCV", "Anomaly Detection"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Nürnberg",
            "remote_policy": "onsite",
            "required_languages": ["German (B2)", "English (B2)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "NLP Engineer / LLM Specialist – Berlin (Hybrid)\n\n"
            "Join our team building next-gen conversational AI products. We're "
            "looking for someone excited about fine-tuning LLMs and building "
            "production RAG pipelines.\n\nRequirements: strong Python, "
            "experience with transformers/HuggingFace, understanding of RAG "
            "and vector databases (FAISS, Pinecone). Bonus: LoRA/QLoRA "
            "fine-tuning experience.\n\nWorking language English, some German "
            "helpful for team meetings. Hybrid, Berlin office 3x/week. We do "
            "sponsor work visas."
        ),
        "label": {
            "job_title": "NLP Engineer / LLM Specialist",
            "company": None,
            "seniority": "unspecified",
            "required_skills": ["Python", "Transformers", "HuggingFace", "RAG", "Vector Databases"],
            "nice_to_have_skills": ["LoRA", "QLoRA"],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Berlin",
            "remote_policy": "hybrid",
            "required_languages": ["English"],
            "visa_sponsorship_mentioned": True,
        },
    },
    {
        "text": (
            "Data Analyst (Junior) – Köln\n\n"
            "Für unser wachsendes Business-Intelligence-Team suchen wir einen "
            "Junior Data Analyst. Du erstellst Reports, Dashboards und "
            "unterstützt Fachbereiche bei datengetriebenen Entscheidungen.\n\n"
            "Wir erwarten: sicherer Umgang mit SQL, erste Erfahrung mit Power "
            "BI oder Tableau, gutes Zahlenverständnis. Deutsch verhandlungssicher.\n\n"
            "Vollzeit, vor Ort in Köln, Einstiegsgehalt ca. 42.000 € brutto/Jahr."
        ),
        "label": {
            "job_title": "Junior Data Analyst",
            "company": None,
            "seniority": "junior",
            "required_skills": ["SQL", "Power BI", "Tableau"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": 42000, "max_eur": 42000, "stated": True},
            "location": "Köln",
            "remote_policy": "onsite",
            "required_languages": ["German (fluent)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "MLOps Engineer (m/w/d) – Remote (EU)\n\n"
            "Wir bauen die Infrastruktur, auf der unsere ML-Modelle laufen. Du "
            "verantwortest CI/CD für ML-Pipelines, Model Monitoring und "
            "Deployment.\n\nRequirements: Docker, Kubernetes, mindestens ein "
            "ML-Tracking-Tool (MLflow, Weights & Biases o.ä.), CI/CD "
            "(GitHub Actions oder GitLab CI), gute Python-Kenntnisse.\n\n"
            "Remote aus der EU möglich. Deutsch nicht erforderlich, Team "
            "spricht Englisch."
        ),
        "label": {
            "job_title": "MLOps Engineer",
            "company": None,
            "seniority": "unspecified",
            "required_skills": ["Docker", "Kubernetes", "MLflow", "CI/CD", "Python"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": None,
            "remote_policy": "remote",
            "required_languages": ["English"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Praktikant KI-Entwicklung (m/w/d) – Augsburg\n\n"
            "Wir suchen einen motivierten Praktikanten im Bereich KI-Entwicklung "
            "für unser kleines Team. Du unterstützt bei der Entwicklung und dem "
            "Testen von ML-Modellen.\n\nErwartet werden: Studium im Bereich "
            "Informatik/KI, Grundkenntnisse Python, Neugier auf neue "
            "Technologien. Deutsch B2 ausreichend.\n\nVergütung: 800 € "
            "monatlich, Vollzeitpraktikum, vor Ort in Augsburg."
        ),
        "label": {
            "job_title": "Praktikant KI-Entwicklung",
            "company": None,
            "seniority": "intern",
            "required_skills": ["Python"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": 9600, "max_eur": 9600, "stated": True},
            "location": "Augsburg",
            "remote_policy": "onsite",
            "required_languages": ["German (B2)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Founding AI Engineer – Munich (Onsite only)\n\n"
            "We're a pre-seed startup (3 people) building an AI copilot for "
            "accountants. Looking for a founding engineer who can move fast "
            "and wear many hats.\n\nWhat you'll need: full-stack comfort, "
            "strong LLM/RAG experience, startup mentality, ok with equity-heavy "
            "comp at this stage.\n\nMust be able to work from our Munich office "
            "daily. No visa sponsorship available at this stage - EU work "
            "authorization required."
        ),
        "label": {
            "job_title": "Founding AI Engineer",
            "company": None,
            "seniority": "senior",
            "required_skills": ["LLM", "RAG", "Full-Stack Development"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Munich",
            "remote_policy": "onsite",
            "required_languages": [],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Data Scientist (m/w/d) – Düsseldorf\n\n"
            "Zur Verstärkung unseres Teams suchen wir einen erfahrenen Data "
            "Scientist mit Schwerpunkt Predictive Analytics.\n\nIhr Profil: "
            "Master in einem quantitativen Fach, 3+ Jahre Berufserfahrung, "
            "sehr gute Kenntnisse in Python und statistischen Methoden, "
            "Erfahrung mit Zeitreihenanalyse wünschenswert.\n\nGehaltsspanne: "
            "62.000 bis 75.000 € brutto p.a. Hybrid-Modell, 3 Tage im Büro "
            "in Düsseldorf."
        ),
        "label": {
            "job_title": "Data Scientist",
            "company": None,
            "seniority": "mid",
            "required_skills": ["Python", "Statistics"],
            "nice_to_have_skills": ["Time Series Analysis"],
            "salary": {"min_eur": 62000, "max_eur": 75000, "stated": True},
            "location": "Düsseldorf",
            "remote_policy": "hybrid",
            "required_languages": [],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Software Engineer – Backend (Python) – Leipzig\n\n"
            "Wir suchen einen Backend-Entwickler für unser Kernprodukt. Kein "
            "ML-Fokus, aber Schnittstellen zu unserem Data-Team.\n\n"
            "Anforderungen: mind. 2 Jahre Erfahrung mit Python (FastAPI/Django), "
            "PostgreSQL, REST-APIs. Deutsch C1 erforderlich, Kundenkontakt "
            "möglich.\n\nVor Ort in Leipzig, kein Homeoffice-Angebot."
        ),
        "label": {
            "job_title": "Software Engineer - Backend (Python)",
            "company": None,
            "seniority": "junior",
            "required_skills": ["Python", "FastAPI", "Django", "PostgreSQL", "REST APIs"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Leipzig",
            "remote_policy": "onsite",
            "required_languages": ["German (C1)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "AI Research Intern – Heidelberg (6 months)\n\n"
            "Join our applied research group working on efficient fine-tuning "
            "methods for small language models. This is a hands-on research "
            "internship, ideal for master's students.\n\nWhat we're looking "
            "for: strong ML fundamentals, PyTorch experience, familiarity with "
            "transformer architectures, some experience with LoRA/PEFT a plus.\n\n"
            "Stipend: 1.200 €/month. English is fine, no German required. "
            "On-site preferred, some remote flexibility possible."
        ),
        "label": {
            "job_title": "AI Research Intern",
            "company": None,
            "seniority": "intern",
            "required_skills": ["PyTorch", "Transformers", "Machine Learning"],
            "nice_to_have_skills": ["LoRA", "PEFT"],
            "salary": {"min_eur": 14400, "max_eur": 14400, "stated": True},
            "location": "Heidelberg",
            "remote_policy": "hybrid",
            "required_languages": ["English"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "IT-Systemadministrator (m/w/d) – Bremen\n\n"
            "Zur Verstärkung unserer IT-Abteilung suchen wir einen "
            "Systemadministrator. Kein KI-/ML-Bezug.\n\nAufgaben: Betreuung "
            "Windows-Server-Landschaft, Netzwerkadministration, First- und "
            "Second-Level-Support.\n\nAnforderungen: abgeschlossene IT-Ausbildung, "
            "Erfahrung mit Windows Server, Active Directory. Deutsch "
            "Muttersprache oder C2.\n\nVor Ort, Bremen, unbefristet."
        ),
        "label": {
            "job_title": "IT-Systemadministrator",
            "company": None,
            "seniority": "unspecified",
            "required_skills": ["Windows Server", "Active Directory", "Network Administration"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Bremen",
            "remote_policy": "onsite",
            "required_languages": ["German (native/C2)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Working Student - Machine Learning (f/m/d) – Berlin\n\n"
            "We're looking for a working student to support our ML team "
            "part-time (up to 20h/week) while you finish your degree.\n\n"
            "You bring: currently enrolled in a CS/Data Science program, solid "
            "Python skills, coursework or projects involving ML. Bonus: "
            "experience with scikit-learn or PyTorch.\n\nCompensation: 18-20 "
            "€/hour. Hybrid - come in 1-2x/week, rest remote. English-speaking "
            "team, German not required."
        ),
        "label": {
            "job_title": "Working Student - Machine Learning",
            "company": None,
            "seniority": "working_student",
            "required_skills": ["Python", "Machine Learning"],
            "nice_to_have_skills": ["scikit-learn", "PyTorch"],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Berlin",
            "remote_policy": "hybrid",
            "required_languages": ["English"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Senior Data Engineer (m/w/d) – Hannover\n\n"
            "Für den Ausbau unserer Dateninfrastruktur suchen wir einen Senior "
            "Data Engineer.\n\nAnforderungen: 5+ Jahre Erfahrung mit "
            "Datenpipelines, sehr gute Kenntnisse in SQL und Python, Erfahrung "
            "mit Airflow oder vergleichbaren Orchestrierungstools, Cloud-"
            "Erfahrung (AWS oder Azure).\n\nGehalt nach Vereinbarung, "
            "wettbewerbsfähig. Standort Hannover, 2 Tage/Woche Homeoffice "
            "möglich. Deutsch verhandlungssicher erforderlich."
        ),
        "label": {
            "job_title": "Senior Data Engineer",
            "company": None,
            "seniority": "senior",
            "required_skills": ["SQL", "Python", "Airflow", "AWS", "Azure"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Hannover",
            "remote_policy": "hybrid",
            "required_languages": ["German (fluent)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Prompt Engineer / AI Product Specialist – Remote (Germany)\n\n"
            "We're building AI-powered products and need someone who deeply "
            "understands how to get the best out of LLMs in production. "
            "Half prompt-engineering, half product thinking.\n\nRequirements: "
            "hands-on experience with GPT-4/Claude/similar APIs, strong "
            "written English, ability to design evaluation frameworks for LLM "
            "outputs. No specific degree required.\n\nFully remote within "
            "Germany. Salary depends on experience, not disclosed publicly."
        ),
        "label": {
            "job_title": "Prompt Engineer / AI Product Specialist",
            "company": None,
            "seniority": "unspecified",
            "required_skills": ["LLM APIs", "Prompt Engineering", "Evaluation Frameworks"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": None,
            "remote_policy": "remote",
            "required_languages": ["English"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Junior Software Developer .NET (m/w/d) – Dresden\n\n"
            "Für unser Entwicklungsteam suchen wir einen Junior .NET-"
            "Entwickler. Kein Data-Science-Bezug.\n\nAnforderungen: Studium "
            "oder Ausbildung im IT-Bereich, Kenntnisse in C# und .NET, erste "
            "praktische Erfahrung von Vorteil. Deutsch fließend.\n\nEinstiegs-"
            "gehalt ca. 45.000 €. Vor Ort in Dresden, gelegentlich Homeoffice "
            "nach Absprache."
        ),
        "label": {
            "job_title": "Junior Software Developer .NET",
            "company": None,
            "seniority": "junior",
            "required_skills": ["C#", ".NET"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": 45000, "max_eur": 45000, "stated": True},
            "location": "Dresden",
            "remote_policy": "onsite",
            "required_languages": ["German (fluent)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Head of AI – Wolfsburg\n\n"
            "Wir suchen eine erfahrene Führungspersönlichkeit für den Aufbau "
            "unserer KI-Abteilung im Automotive-Umfeld.\n\nAnforderungen: 8+ "
            "Jahre Erfahrung in KI/ML, davon mehrere Jahre in "
            "Führungsverantwortung, tiefes technisches Verständnis (nicht nur "
            "Management), Branchenerfahrung Automotive von Vorteil. Deutsch "
            "und Englisch verhandlungssicher.\n\nGehalt: 130.000 - 160.000 € "
            "zzgl. Bonus. Standort Wolfsburg, hybrides Arbeiten möglich."
        ),
        "label": {
            "job_title": "Head of AI",
            "company": None,
            "seniority": "lead",
            "required_skills": ["Machine Learning", "AI", "Team Leadership"],
            "nice_to_have_skills": ["Automotive Industry Experience"],
            "salary": {"min_eur": 130000, "max_eur": 160000, "stated": True},
            "location": "Wolfsburg",
            "remote_policy": "hybrid",
            "required_languages": ["German (fluent)", "English (fluent)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Data Science Bootcamp Graduate wanted – Junior role – Essen\n\n"
            "We welcome career changers! Just finished a data science bootcamp "
            "or self-taught? We'll train you further on the job.\n\nWhat "
            "matters: solid Python basics, understanding of core ML concepts, "
            "genuine curiosity. A portfolio of personal projects counts more "
            "than a specific degree.\n\nGerman B1 minimum (client-facing "
            "role), English helpful. Essen office, hybrid after onboarding."
        ),
        "label": {
            "job_title": "Junior Data Scientist (Career Changer)",
            "company": None,
            "seniority": "junior",
            "required_skills": ["Python", "Machine Learning"],
            "nice_to_have_skills": ["English"],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Essen",
            "remote_policy": "hybrid",
            "required_languages": ["German (B1)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Research Engineer, Foundation Models – Berlin/Remote\n\n"
            "Join our research lab working on efficient training methods for "
            "foundation models. PhD not required if you have equivalent "
            "practical experience.\n\nRequirements: deep understanding of "
            "transformer architectures, experience training models at scale, "
            "publication record a plus but not required, strong software "
            "engineering skills (this isn't just research, you'll ship code).\n\n"
            "Salary: 85.000 - 110.000 € depending on experience. Remote-"
            "friendly, Berlin office available. We sponsor visas."
        ),
        "label": {
            "job_title": "Research Engineer, Foundation Models",
            "company": None,
            "seniority": "senior",
            "required_skills": ["Transformers", "Model Training", "Software Engineering"],
            "nice_to_have_skills": ["Publications"],
            "salary": {"min_eur": 85000, "max_eur": 110000, "stated": True},
            "location": "Berlin",
            "remote_policy": "remote",
            "required_languages": [],
            "visa_sponsorship_mentioned": True,
        },
    },
    {
        "text": (
            "Vertriebsmitarbeiter (m/w/d) Außendienst – Region Bayern\n\n"
            "Wir suchen einen erfahrenen Vertriebsmitarbeiter für unseren "
            "Außendienst in Bayern. Kein IT-/Tech-Bezug.\n\nAnforderungen: "
            "abgeschlossene kaufmännische Ausbildung, mehrjährige "
            "Vertriebserfahrung, Führerschein Klasse B. Deutsch Muttersprache.\n\n"
            "Fixgehalt plus Provision, Firmenwagen auch zur privaten Nutzung. "
            "Reisetätigkeit innerhalb Bayerns."
        ),
        "label": {
            "job_title": "Vertriebsmitarbeiter Außendienst",
            "company": None,
            "seniority": "unspecified",
            "required_skills": ["Sales Experience", "Driver's License Class B"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Bayern",
            "remote_policy": "onsite",
            "required_languages": ["German (native)"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Applied Scientist - Computer Vision (all genders) – München\n\n"
            "Wir suchen einen Applied Scientist im Bereich Computer Vision für "
            "unser Forschungsteam im industriellen Umfeld.\n\nAnforderungen: "
            "PhD oder gleichwertige Erfahrung in Computer Vision, "
            "Publikationen in relevanten Konferenzen (CVPR, ICCV, ECCV) von "
            "Vorteil, sehr gute PyTorch-Kenntnisse, Erfahrung mit Anomaly "
            "Detection oder Segmentation.\n\nGehalt: 78.000 - 95.000 €. "
            "München, hybrides Modell, Englisch als Arbeitssprache ausreichend."
        ),
        "label": {
            "job_title": "Applied Scientist - Computer Vision",
            "company": None,
            "seniority": "senior",
            "required_skills": ["Computer Vision", "PyTorch", "Anomaly Detection", "Segmentation"],
            "nice_to_have_skills": ["Publications (CVPR/ICCV/ECCV)"],
            "salary": {"min_eur": 78000, "max_eur": 95000, "stated": True},
            "location": "München",
            "remote_policy": "hybrid",
            "required_languages": ["English"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Junior AI Engineer (m/w/d) – Frankfurt\n\n"
            "Einstiegsposition für Absolventen mit erster praktischer "
            "Erfahrung im KI-Bereich (Praktikum, Abschlussarbeit, eigene "
            "Projekte zählen).\n\nAnforderungen: Bachelor oder Master in "
            "Informatik, KI oder verwandtem Fach, gute Python-Kenntnisse, "
            "Grundverständnis von Machine Learning und Deep Learning. Deutsch "
            "und Englisch gut in Wort und Schrift.\n\nEinstiegsgehalt: 50.000 - "
            "56.000 €. Frankfurt, hybrides Arbeiten (3 Tage Büro)."
        ),
        "label": {
            "job_title": "Junior AI Engineer",
            "company": None,
            "seniority": "junior",
            "required_skills": ["Python", "Machine Learning", "Deep Learning"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": 50000, "max_eur": 56000, "stated": True},
            "location": "Frankfurt",
            "remote_policy": "hybrid",
            "required_languages": ["German", "English"],
            "visa_sponsorship_mentioned": False,
        },
    },
    {
        "text": (
            "Elektroniker für Betriebstechnik (m/w/d) – Duisburg\n\n"
            "Für unser Instandhaltungsteam suchen wir einen Elektroniker für "
            "Betriebstechnik. Kein Software-/IT-Bezug.\n\nAnforderungen: "
            "abgeschlossene Ausbildung als Elektroniker für Betriebstechnik "
            "oder vergleichbar, Schichtbereitschaft. Deutsch erforderlich für "
            "Sicherheitsunterweisungen.\n\nTarifgehalt nach IG Metall, Standort "
            "Duisburg, Schichtarbeit."
        ),
        "label": {
            "job_title": "Elektroniker für Betriebstechnik",
            "company": None,
            "seniority": "unspecified",
            "required_skills": ["Electrical Maintenance", "Shift Work Availability"],
            "nice_to_have_skills": [],
            "salary": {"min_eur": None, "max_eur": None, "stated": False},
            "location": "Duisburg",
            "remote_policy": "onsite",
            "required_languages": ["German"],
            "visa_sponsorship_mentioned": False,
        },
    },
]


def main() -> None:
    validated = []
    for i, ex in enumerate(EXAMPLES):
        try:
            JobExtraction.model_validate(ex["label"])
        except Exception as exc:
            raise ValueError(f"Example {i} failed schema validation: {exc}") from exc
        validated.append(ex)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in validated:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(validated)} validated examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
