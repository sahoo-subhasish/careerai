from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import io
import google.generativeai as genai

app = FastAPI(title="Career Path Optimizer API")

GEMINI_API_KEY = "AIzaSyAC1tTupH1zX2T4nVSZraQP3sH5Qfn-850"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

SKILLS_TAXONOMY = {
    "programming": ["Python", "JavaScript", "Java", "C++", "TypeScript", "Go", "Rust", "PHP", "Ruby", "Swift"],
    "web": ["React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "FastAPI", "Express", "Next.js", "HTML", "CSS"],
    "data": ["Machine Learning", "Deep Learning", "Data Analysis", "SQL", "MongoDB", "Pandas", "NumPy", "TensorFlow", "PyTorch", "Scikit-learn"],
    "cloud": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins", "CI/CD"],
    "tools": ["Git", "GitHub", "GitLab", "Jira", "VS Code", "Linux", "Bash"],
    "soft_skills": ["Communication", "Leadership", "Problem Solving", "Teamwork", "Agile", "Scrum"]
}

ALL_SKILLS = [skill for category in SKILLS_TAXONOMY.values() for skill in category]

JOB_ROLES = {
    "Full Stack Developer": {
        "required": ["JavaScript", "React", "Node.js", "SQL", "Git", "HTML", "CSS"],
        "preferred": ["TypeScript", "Docker", "AWS", "MongoDB", "Next.js"],
        "importance": {
            "JavaScript": 10, "React": 9, "Node.js": 9, "SQL": 8, "Git": 7,
            "TypeScript": 8, "Docker": 7, "AWS": 7, "HTML": 9, "CSS": 8
        }
    },
    "Data Scientist": {
        "required": ["Python", "Machine Learning", "Data Analysis", "SQL", "Pandas"],
        "preferred": ["Deep Learning", "AWS", "Docker", "NumPy", "TensorFlow"],
        "importance": {
            "Python": 10, "Machine Learning": 10, "Data Analysis": 9, "SQL": 8,
            "Pandas": 8, "Deep Learning": 9, "NumPy": 7, "TensorFlow": 8
        }
    },
    "DevOps Engineer": {
        "required": ["Docker", "Kubernetes", "AWS", "Git", "Python", "Linux"],
        "preferred": ["Terraform", "Jenkins", "Azure", "Go", "CI/CD"],
        "importance": {
            "Docker": 10, "Kubernetes": 9, "AWS": 9, "Git": 8, "Python": 7,
            "Terraform": 8, "Jenkins": 7, "Linux": 9, "CI/CD": 8
        }
    },
    "Backend Developer": {
        "required": ["Python", "SQL", "FastAPI", "Git", "Docker"],
        "preferred": ["PostgreSQL", "MongoDB", "Redis", "AWS", "Kubernetes"],
        "importance": {
            "Python": 10, "SQL": 9, "FastAPI": 8, "Git": 8, "Docker": 8,
            "PostgreSQL": 7, "MongoDB": 6, "AWS": 7
        }
    },
    "Frontend Developer": {
        "required": ["JavaScript", "React", "HTML", "CSS", "Git"],
        "preferred": ["TypeScript", "Next.js", "Vue.js", "Tailwind CSS"],
        "importance": {
            "JavaScript": 10, "React": 9, "HTML": 9, "CSS": 9, "Git": 7,
            "TypeScript": 8, "Next.js": 7
        }
    }
}

COURSES = [
    {"id": 1, "name": "React & TypeScript - The Practical Guide", "skills": ["React", "TypeScript"], "duration": 7.5, "rating": 4.7, "price": 399, "url": "https://www.udemy.com/course/react-typescript-the-practical-guide/?couponCode=CP250105G1"},
    {"id": 2, "name": "The Complete Full-Stack Web Development Bootcamp", "skills": ["HTML","CSS","PostgreSQL", "JavaScript", "Node.js", "React", "Web3", "DApps"], "duration": 61, "rating": 4.7, "price": 399, "url": "https://www.udemy.com/course/the-complete-web-development-bootcamp/?couponCode=CP250105G1"},
    {"id": 3, "name": "Docker & Kubernetes: The Practical Guide [2025 Edition]", "skills": ["Docker", "Kubernetes", "CI/CD"], "duration": 23.5, "rating": 4.7, "price": 459, "url": "https://www.udemy.com/course/docker-kubernetes-the-practical-guide/?couponCode=CP250105G1"},
    {"id": 4, "name": "Machine Learning A-Z: AI, Python & R + ChatGPT Prize [2026]", "skills": ["Machine Learning", "Python", "NumPy", "Scikit-learn"], "duration": 42.5, "rating": 4.5, "price": 399, "url": "https://www.udemy.com/course/machinelearning/?couponCode=CP250105G1"},
    {"id": 5, "name": "Ultimate AWS Certified Solutions Architect Professional 2026", "skills": ["AWS", "Cloud Computing", "Docker"], "duration": 16.5, "rating": 4.7, "price": 409, "url": "https://www.udemy.com/course/aws-solutions-architect-professional/?couponCode=CP250105G1"},
    {"id": 6, "name": "SQL for Data Analysis: Advanced SQL Querying Techniques", "skills": ["SQL", "Data Analysis", "PostgreSQL"], "duration": 8.5, "rating": 4.7, "price": 399, "url": "https://www.udemy.com/course/sql-advanced-queries/?couponCode=CP250105G1"},
    {"id": 7, "name": "MongoDB - The Complete Developer's Guide 2025", "skills": ["MongoDB", "Node.js", "Express"], "duration": 17.5, "rating": 4.6, "price": 399, "url": "https://www.udemy.com/course/mongodb-the-complete-developers-guide/?couponCode=CP250105G1"},
    {"id": 8, "name": "A deep understanding of deep learning (with Python intro)", "skills": ["Deep Learning", "Python", "TensorFlow", "PyTorch"], "duration": 57.5, "rating": 4.8, "price": 399, "url": "https://www.udemy.com/course/deeplearning_x/?kw=A+deep+understanding+of+deep+learning+%28with+Python+intro%29&src=sac&couponCode=CP250105G1"},
    {"id": 9, "name": "Complete Backend Development 2025 Bundle - Python Roadmap", "skills": ["Python", "FastAPI", "Django", "SQL"], "duration": 21, "rating": 4.2, "price": 399, "url": "https://www.udemy.com/course/software-developer-masterclass/?couponCode=CP250105G1"},
    {"id": 10, "name": "Decoding DevOps – From Basics to Advanced Projects with AI", "skills": ["Linux", "Bash", "Git", "Jenkins"], "duration": 64, "rating": 4.6, "price": 399, "url": "https://www.udemy.com/course/decodingdevops/?couponCode=CP250105G1"},
]

TECHNICAL_SKILLS_HEADINGS = [
    r"technical\s+skills?",
    r"skills?",
    r"core\s+skills?",
    r"key\s+skills?",
    r"professional\s+skills?",
    r"technologies?",
    r"technical\s+competencies",
    r"technical\s+expertise",
    r"tech\s+stack",
    r"technology\s+stack",
    r"programming\s+skills?",
    r"programming\s+languages?",
    r"languages?\s+and\s+technologies",
    r"languages?\s+&\s+technologies",

    r"tools?\s+and\s+technologies",
    r"tools?\s+&\s+technologies",
    r"frameworks?\s+and\s+tools?",
    r"technical\s+tools?",
    r"technical\s+proficiencies",
    r"areas?\s+of\s+expertise",
    r"core\s+competencies",
    r"competencies",
    r"software\s+skills?",
    r"it\s+skills?",
    r"computer\s+skills?",
    r"technical\s+skillset",
    r"skillset",
]
SECTION_STOP_MARKERS = [
    r"experience",
    r"work\s+experience",
    r"professional\s+experience",
    r"employment\s+history",
    r"education",
    r"academic\s+background",
    r"certifications?",
    r"projects?",
    r"achievements?",
    r"publications?",
    r"references?",
    r"hobbies",
    r"interests?",
    r"languages?",  # spoken languages section
    r"extracurricular\s+activities?",
    r"summary",
    r"objective",
    r"profile",
]
