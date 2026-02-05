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

def parse_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing error: {str(e)}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'[^\w\s.,;:()\-+/#+\n]', '', text)
    

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

def extract_technical_skills_section(resume_text: str) -> Tuple[str, str]:
    
    skills_section, method = extract_section_by_patterns(resume_text)
    if skills_section and len(skills_section.strip()) > 20:
        return skills_section, method
    
    skills_section, method = extract_section_by_structure(resume_text)
    if skills_section and len(skills_section.strip()) > 20:
        return skills_section, method
    
    skills_section, method = extract_section_by_ai(resume_text)
    if skills_section and len(skills_section.strip()) > 20:
        return skills_section, method
    
    return resume_text, "fallback_full_text"


def extract_section_by_patterns(text: str) -> Tuple[str, str]:
    lines = text.split('\n')
    skills_start_idx = -1
    skills_end_idx = len(lines)
    matched_heading = ""
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        for heading_pattern in TECHNICAL_SKILLS_HEADINGS:
            if re.match(r'^\s*' + heading_pattern + r'\s*[:]*\s*$', line_lower):
                skills_start_idx = i
                matched_heading = line.strip()
                break
        if skills_start_idx != -1:
            break
    
    if skills_start_idx == -1:
        return "", "pattern_not_found"
    
    for i in range(skills_start_idx + 1, len(lines)):
        line_lower = lines[i].lower().strip()
        
        for stop_pattern in SECTION_STOP_MARKERS:
            if re.match(r'^\s*' + stop_pattern + r'\s*[:]*\s*$', line_lower):
                skills_end_idx = i
                break
        if skills_end_idx != len(lines):
            break
    
    skills_section = '\n'.join(lines[skills_start_idx + 1:skills_end_idx])
    
    return skills_section.strip(), f"pattern_match:{matched_heading}"


def extract_section_by_structure(text: str) -> Tuple[str, str]:
    lines = text.split('\n')
    potential_skills_blocks = []
    current_block = []
    in_skills_block = False
    for line in lines:
        stripped = line.strip()
        
        is_skill_line = bool(re.match(r'^[\•\-\*\→]', stripped)) or \
                       ',' in stripped or \
                       '|' in stripped or \
                       re.search(r'\b(python|java|javascript|sql|aws|docker|react)\b', stripped.lower())
        
        if is_skill_line:
            in_skills_block = True
            current_block.append(line)
        elif in_skills_block:
            if current_block:
                potential_skills_blocks.append('\n'.join(current_block))
                current_block = []
            in_skills_block = False
    
    if current_block:
        potential_skills_blocks.append('\n'.join(current_block))
    
    if potential_skills_blocks:
        largest_block = max(potential_skills_blocks, key=len)
        if len(largest_block) > 50:  # Minimum threshold
            return largest_block, "structure_based_extraction"
    
    return "", "structure_not_found"


def extract_section_by_ai(text: str) -> Tuple[str, str]:
    try:
        prompt = f"""You are an expert resume parser. Extract ONLY the technical skills section from this resume.

Resume Text:
{text[:3000]}
Instructions:
- Find the section containing technical skills (programming languages, frameworks, tools, technologies)
- Extract ONLY that section's content
- Do NOT include work experience, education, or other sections
- If multiple skill sections exist, combine them
- Return ONLY the extracted skills text, nothing else
- If no clear skills section exists, return "NOT_FOUND"

Technical Skills Section:"""

        response = gemini_model.generate_content(prompt)
        extracted_text = response.text.strip()
        
        if extracted_text and extracted_text != "NOT_FOUND" and len(extracted_text) > 20:
            return extracted_text, "ai_assisted_extraction"
        
    except Exception as e:
        print(f"AI extraction error: {str(e)}")
    return "", "ai_extraction_failed"

def parse_pdf_with_skills_extraction(file_content: bytes) -> Dict[str, str]:
    try:
        raw_text = parse_pdf(file_content)
        
        # Step 2: Clean the text
        cleaned_text = clean_text(raw_text)
        
        skills_section, extraction_method = extract_technical_skills_section(cleaned_text)
        return {
            "full_text": cleaned_text,
            "skills_section": skills_section,
            "extraction_method": extraction_method,
            "skills_section_length": len(skills_section)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing error: {str(e)}")


def extract_skills(text: str, use_skills_section_only: bool = True) -> List[str]:
    found_skills = []
    text_lower = text.lower()
    
    for skill in ALL_SKILLS:
        # Create word boundary pattern
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    # Only apply version-based inference if NOT in strict section-only mode
    if use_skills_section_only:
        # Strict mode: return only exact skill name matches
        return list(set(found_skills))
    
    enhanced_skills = found_skills.copy()
    
    
    version_patterns = [
        r'\b(python|java|node\.?js|react|angular|vue)\s*[\d.]+\b',
        r'\b(aws|azure|gcp)\s+(certified|associate|professional)\b',
    ]
    for pattern in version_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            skill_name = match.group(1).title()
            
            if 'node' in skill_name.lower():
                skill_name = 'Node.js'
            if skill_name in ALL_SKILLS and skill_name not in enhanced_skills:
                enhanced_skills.append(skill_name)
    
    return list(set(enhanced_skills))  # Remove duplicates

def get_embeddings(texts: List[str]) -> np.ndarray:
    return sbert_model.encode(texts, convert_to_numpy=True)


def normalize(skill: str) -> str:
    return skill.strip().lower()


def calculate_skill_similarity(user_skills: List[str], target_skill: str) -> float:
    
    if target_skill in user_skills:
        return 1.0
    
    if not user_skills:
        return 0.0

    target_embedding = get_embeddings([target_skill])
    user_embeddings = get_embeddings(user_skills)

    similarities = cosine_similarity(target_embedding, user_embeddings)[0]
    return float(np.max(similarities))

    def analyze_skill_gaps(user_skills: List[str], role: str) -> Dict:
    if role not in JOB_ROLES:
        raise HTTPException(status_code=404, detail=f"Role '{role}' not found")

    role_data = JOB_ROLES[role]
    required_skills = role_data["required"]
    preferred_skills = role_data.get("preferred", [])
    importance = role_data.get("importance", {})

    all_target_skills = list(set(required_skills + preferred_skills))
    skill_statuses = []
    total_score = 0
    max_possible_score = 0

    for skill in all_target_skills:
        similarity = calculate_skill_similarity(user_skills, skill)
        weight = importance.get(skill, 5)
        
        if similarity >= 0.95:
            status = "strong"
            score = weight * 1.0
        elif similarity >= 0.7:
            status = "partial"
            score = weight * 0.6
        else:
            status = "missing"
            score = 0
        total_score += score
        max_possible_score += weight
        is_required = skill in required_skills
        
        skill_statuses.append({
            "skill": skill,
            "similarity": round(similarity, 3),
            "status": status,
            "required": is_required,
            "weight": weight
        })
    
    # Calculate readiness percentage
    readiness = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
    
    # Sort by importance
    skill_statuses.sort(key=lambda x: x["weight"], reverse=True)
    
    return {
        "role": role,
        "readiness_percentage": round(readiness, 1),
        "skills": skill_statuses,
        "summary": {
            "strong": sum(1 for s in skill_statuses if s["status"] == "strong"),
            "partial": sum(1 for s in skill_statuses if s["status"] == "partial"),
            "missing": sum(1 for s in skill_statuses if s["status"] == "missing")
        }
    }

# ============================================================================
# MODULE 5: COURSE RECOMMENDATION ENGINE
# ============================================================================

def recommend_courses(gap_analysis: Dict, max_recommendations: int = 5) -> List[Dict]:
    """
    Recommend courses based on skill gaps
    Prioritizes: missing required skills > partial skills > preferred skills
    """
    missing_skills = [s["skill"] for s in gap_analysis["skills"] if s["status"] == "missing" and s["required"]]
    partial_skills = [s["skill"] for s in gap_analysis["skills"] if s["status"] == "partial"]
    
    priority_skills = missing_skills + partial_skills
    
    course_scores = []
    
    for course in COURSES:
        course_skills = set([s.lower() for s in course["skills"]])
        priority_skills_lower = set([s.lower() for s in priority_skills])
        
        # Calculate overlap
        overlap = len(course_skills.intersection(priority_skills_lower))
        
        if overlap > 0:
            # Score = overlap * rating * (1/price_factor)
            price_factor = course["price"] / 400  # Normalize around 400
            score = overlap * course["rating"] / price_factor
            
            course_scores.append({
                "course": course,
                "score": score,
                "relevant_skills": list(course_skills.intersection(priority_skills_lower))
            })
    
    # Sort by score
    course_scores.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top recommendations
    recommendations = []
    for item in course_scores[:max_recommendations]:
        course = item["course"]
        recommendations.append({
            "id": course["id"],
            "name": course["name"],
            "skills": course["skills"],
            "duration": course["duration"],
            "rating": course["rating"],
            "price": course["price"],
            "url": course["url"],
            "relevance_score": round(item["score"], 2),
            "addresses_gaps": item["relevant_skills"]
        })
    
    return recommendations


