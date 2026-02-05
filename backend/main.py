# FastAPI + spaCy + SBERT + Deterministic Logic + Enhanced PDF Extraction

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

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initializing SBERT Model (loaded once at startup)
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

# ============================================================================
# MODULE 1: ENHANCED RESUME PARSING WITH INTELLIGENT SECTION EXTRACTION
# ============================================================================

# Comprehensive list of technical skills section heading variations
TECHNICAL_SKILLS_HEADINGS = [
    # Standard variations
    r"technical\s+skills?",
    r"skills?",
    r"core\s+skills?",
    r"key\s+skills?",
    r"professional\s+skills?",
    
    # Technology-focused
    r"technologies?",
    r"technical\s+competencies",
    r"technical\s+expertise",
    r"tech\s+stack",
    r"technology\s+stack",
    
    # Programming-focused
    r"programming\s+skills?",
    r"programming\s+languages?",
    r"languages?\s+and\s+technologies",
    r"languages?\s+&\s+technologies",
    
    # Tools and frameworks
    r"tools?\s+and\s+technologies",
    r"tools?\s+&\s+technologies",
    r"frameworks?\s+and\s+tools?",
    r"technical\s+tools?",
    
    # Competencies
    r"technical\s+proficiencies",
    r"areas?\s+of\s+expertise",
    r"core\s+competencies",
    r"competencies",
    
    # Other variations
    r"software\s+skills?",
    r"it\s+skills?",
    r"computer\s+skills?",
    r"technical\s+skillset",
    r"skillset",
]

# Section headings that should STOP skill extraction (next section markers)
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
    """Parse PDF and extract raw text"""
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
    """Clean text while preserving structure for section detection"""
    if not text:
        return ""
    
    # Remove special characters but keep structure
    text = re.sub(r'[^\w\s.,;:()\-+/#+\n]', '', text)
    
    # Normalize excessive whitespace but preserve line breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()


def extract_technical_skills_section(resume_text: str) -> Tuple[str, str]:
    """
    Intelligently extract the technical skills section from resume.
    Returns: (skills_section_text, extraction_method)
    
    Uses multiple strategies:
    1. Pattern-based section detection (primary)
    2. Structure-based extraction (secondary)
    3. AI-assisted extraction (fallback)
    """
    
    # Strategy 1: Pattern-based section detection
    skills_section, method = extract_section_by_patterns(resume_text)
    if skills_section and len(skills_section.strip()) > 20:
        return skills_section, method
    
    # Strategy 2: Structure-based extraction (look for bullet points, lists)
    skills_section, method = extract_section_by_structure(resume_text)
    if skills_section and len(skills_section.strip()) > 20:
        return skills_section, method
    
    # Strategy 3: AI-assisted extraction (fallback)
    skills_section, method = extract_section_by_ai(resume_text)
    if skills_section and len(skills_section.strip()) > 20:
        return skills_section, method
    
    # Final fallback: return full text
    return resume_text, "fallback_full_text"


def extract_section_by_patterns(text: str) -> Tuple[str, str]:
    """Extract skills section using heading pattern matching"""
    
    lines = text.split('\n')
    skills_start_idx = -1
    skills_end_idx = len(lines)
    matched_heading = ""
    
    # Find skills section start
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Check if line matches any technical skills heading
        for heading_pattern in TECHNICAL_SKILLS_HEADINGS:
            if re.match(r'^\s*' + heading_pattern + r'\s*[:]*\s*$', line_lower):
                skills_start_idx = i
                matched_heading = line.strip()
                break
        
        if skills_start_idx != -1:
            break
    
    # If no skills section found, return empty
    if skills_start_idx == -1:
        return "", "pattern_not_found"
    
    # Find skills section end (next major section)
    for i in range(skills_start_idx + 1, len(lines)):
        line_lower = lines[i].lower().strip()
        
        # Check if this line is a stop marker (next section)
        for stop_pattern in SECTION_STOP_MARKERS:
            if re.match(r'^\s*' + stop_pattern + r'\s*[:]*\s*$', line_lower):
                skills_end_idx = i
                break
        
        if skills_end_idx != len(lines):
            break
    
    # Extract the skills section
    skills_section = '\n'.join(lines[skills_start_idx + 1:skills_end_idx])
    
    return skills_section.strip(), f"pattern_match:{matched_heading}"


def extract_section_by_structure(text: str) -> Tuple[str, str]:
    """Extract skills by detecting bullet points and structured lists"""
    
    lines = text.split('\n')
    potential_skills_blocks = []
    current_block = []
    in_skills_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Detect lines that look like skill entries (bullet points, commas, pipes)
        is_skill_line = bool(re.match(r'^[\•\-\*\→]', stripped)) or \
                       ',' in stripped or \
                       '|' in stripped or \
                       re.search(r'\b(python|java|javascript|sql|aws|docker|react)\b', stripped.lower())
        
        if is_skill_line:
            in_skills_block = True
            current_block.append(line)
        elif in_skills_block:
            # End of skills block
            if current_block:
                potential_skills_blocks.append('\n'.join(current_block))
                current_block = []
            in_skills_block = False
    
    # Add final block if exists
    if current_block:
        potential_skills_blocks.append('\n'.join(current_block))
    
    # Return the largest block (most likely the skills section)
    if potential_skills_blocks:
        largest_block = max(potential_skills_blocks, key=len)
        if len(largest_block) > 50:  # Minimum threshold
            return largest_block, "structure_based_extraction"
    
    return "", "structure_not_found"


def extract_section_by_ai(text: str) -> Tuple[str, str]:
    """Use Gemini AI to intelligently extract technical skills section"""
    
    try:
        prompt = f"""You are an expert resume parser. Extract ONLY the technical skills section from this resume.

Resume Text:
{text[:3000]}  # Limit to first 3000 chars for efficiency

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
    """
    Enhanced PDF parsing that extracts both full text and technical skills section
    """
    try:
        # Step 1: Extract raw text from PDF
        raw_text = parse_pdf(file_content)
        
        # Step 2: Clean the text
        cleaned_text = clean_text(raw_text)
        
        # Step 3: Extract technical skills section
        skills_section, extraction_method = extract_technical_skills_section(cleaned_text)
        
        return {
            "full_text": cleaned_text,
            "skills_section": skills_section,
            "extraction_method": extraction_method,
            "skills_section_length": len(skills_section)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing error: {str(e)}")


# ============================================================================
# MODULE 2: ENHANCED SKILL EXTRACTION (from skills section only)
# ============================================================================

def extract_skills(text: str, use_skills_section_only: bool = True) -> List[str]:
    """
    Extract skills from text with optional focus on skills section
    
    Args:
        text: Either full resume text or extracted skills section
        use_skills_section_only: If True, extract ONLY exact explicit skill name matches (no version inference)
    """
    found_skills = []
    text_lower = text.lower()
    
    # Enhanced pattern matching for skills
    for skill in ALL_SKILLS:
        # Create word boundary pattern
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    # Only apply version-based inference if NOT in strict section-only mode
    if use_skills_section_only:
        # Strict mode: return only exact skill name matches
        return list(set(found_skills))
    
    # Additional extraction: look for version numbers and frameworks
    # e.g., "Python 3.x", "React 18", "Node.js 16"
    enhanced_skills = found_skills.copy()
    
    # Look for skills with versions or specifications
    version_patterns = [
        r'\b(python|java|node\.?js|react|angular|vue)\s*[\d.]+\b',
        r'\b(aws|azure|gcp)\s+(certified|associate|professional)\b',
    ]
    
    for pattern in version_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            skill_name = match.group(1).title()
            # Normalize specific cases
            if 'node' in skill_name.lower():
                skill_name = 'Node.js'
            if skill_name in ALL_SKILLS and skill_name not in enhanced_skills:
                enhanced_skills.append(skill_name)
    
    return list(set(enhanced_skills))  # Remove duplicates


# ============================================================================
# MODULE 3: EMBEDDING MODULE (SBERT - Black Box Encoder)
# ============================================================================

def get_embeddings(texts: List[str]) -> np.ndarray:
    """Convert text to semantic vectors using SBERT"""
    return sbert_model.encode(texts, convert_to_numpy=True)


# ============================================================================
# MODULE 4: SIMILARITY & GAP ANALYSIS (Deterministic Logic)
# ============================================================================

def normalize(skill: str) -> str:
    return skill.strip().lower()


def calculate_skill_similarity(user_skills: List[str], target_skill: str) -> float:
    """Calculate similarity between user skills and target skill"""
    # HARD RULE: explicit resume claim = full proficiency
    if target_skill in user_skills:
        return 1.0

    # Otherwise, allow semantic inference
    if not user_skills:
        return 0.0

    target_embedding = get_embeddings([target_skill])
    user_embeddings = get_embeddings(user_skills)

    similarities = cosine_similarity(target_embedding, user_embeddings)[0]
    return float(np.max(similarities))


def analyze_skill_gaps(user_skills: List[str], role: str) -> Dict:
    """
    Deterministic gap analysis with explicit thresholds
    NO LLM INVOLVEMENT - Pure logic
    """
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
        
        # Deterministic thresholds
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


# ============================================================================
# MODULE 6: AI CAREER COUNSELOR (Gemini Integration)
# ============================================================================

def generate_ai_counselor_advice(gap_analysis: Dict, user_skills: List[str], recommendations: List[Dict]) -> Dict:
    """
    Generate personalized career counseling using Gemini AI
    """
    try:
        role = gap_analysis["role"]
        readiness = gap_analysis["readiness_percentage"]
        
        strong_skills = [s["skill"] for s in gap_analysis["skills"] if s["status"] == "strong"]
        partial_skills = [s["skill"] for s in gap_analysis["skills"] if s["status"] == "partial"]
        missing_skills = [s["skill"] for s in gap_analysis["skills"] if s["status"] == "missing"]
        
        prompt = f"""You are an expert career counselor helping someone transition to a {role} role.

Current Status:
- Role Readiness: {readiness}%
- Strong Skills: {', '.join(strong_skills[:5]) if strong_skills else 'None identified'}
- Partial Skills: {', '.join(partial_skills[:5]) if partial_skills else 'None'}
- Missing Critical Skills: {', '.join(missing_skills[:5]) if missing_skills else 'None'}

Recommended Courses:
{json.dumps([{"name": r["name"], "skills": r["skills"]} for r in recommendations[:3]], indent=2)}

Provide:
1. A personalized, encouraging counseling message (2-3 paragraphs)
2. A 6-month learning roadmap divided into 3 phases (2 months each)

Format your response as JSON:
{{
    "counseling": "Your personalized message here...",
    "roadmap": {{
        "month1_2": {{
            "title": "Phase 1 Title",
            "description": "What to focus on",
            "focus_skills": ["skill1", "skill2", "skill3"]
        }},
        "month3_4": {{
            "title": "Phase 2 Title",
            "description": "What to focus on",
            "focus_skills": ["skill1", "skill2"]
        }},
        "month5_6": {{
            "title": "Phase 3 Title",
            "description": "What to focus on",
            "focus_skills": ["skill1", "skill2"]
        }}
    }}
}}

Be specific, actionable, and motivating. Reference their current strengths."""

        response = gemini_model.generate_content(prompt)
        response_text = ""
        for part in response.parts:
            if hasattr(part, 'text'):
                response_text += part.text
        response_text = response_text.strip()
        
        # Try to extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            result = {
                "counseling": response_text if response_text else f"I've analyzed your profile for {role}. You have a strong foundation with skills like {', '.join(strong_skills[:3]) if strong_skills else 'several key areas'}, but there are critical gaps in {', '.join(missing_skills[:3]) if missing_skills else 'some areas'}. Focus on the recommended courses to close these gaps and build practical projects to demonstrate your skills.",
                "roadmap": {
                    "month1_2": {"title": "Foundations & Critical Gaps", "description": "Focus on the most critical missing skills", "focus_skills": missing_skills[:3] if missing_skills else []},
                    "month3_4": {"title": "Intermediate Skills & Projects", "description": "Strengthen partial skills and build projects", "focus_skills": partial_skills[:3] if partial_skills else []},
                    "month5_6": {"title": "Job Readiness & Portfolio", "description": "Complete portfolio projects and interview prep", "focus_skills": []}
                }
            }
        
        return result
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        # Fallback response
        strong_skills = [s["skill"] for s in gap_analysis["skills"] if s["status"] == "strong"]
        missing_skills = [s["skill"] for s in gap_analysis["skills"] if s["status"] == "missing"]
        return {
            "counseling": f"I've analyzed your profile for {gap_analysis['role']}. You have a strong foundation in {', '.join(strong_skills[:3]) if strong_skills else 'several areas'}, but there are critical gaps in {', '.join(missing_skills[:3]) if missing_skills else 'key areas'}. Focus on the recommended courses to close these gaps and build practical projects.",
            "roadmap": {
                "month1_2": {"title": "Foundations & Gaps", "description": "Focus on critical gaps", "focus_skills": missing_skills[:3] if missing_skills else []},
                "month3_4": {"title": "Advanced Skills", "description": "Build practical expertise", "focus_skills": []},
                "month5_6": {"title": "Job Ready", "description": "Portfolio and interview prep", "focus_skills": []}
            }
        }


# ============================================================================
# API MODELS
# ============================================================================

class ResumeText(BaseModel):
    text: str

class AnalysisRequest(BaseModel):
    skills: List[str]
    role: str

class RoadmapRequest(BaseModel):
    user_skills: List[str]
    role: str
    gap_analysis: Dict
    recommendations: List[Dict]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "Career Path Optimizer API - Enhanced Edition",
        "version": "2.0",
        "features": [
            "Intelligent technical skills section extraction",
            "Multi-strategy PDF parsing",
            "AI-assisted skill detection"
        ],
        "endpoints": [
            "/parse-resume-pdf",
            "/parse-resume-enhanced",
            "/extract-skills",
            "/analyze-gaps",
            "/recommend-courses",
            "/analyze-resume",
            "/available-roles"
        ]
    }


@app.post("/parse-resume-pdf")
async def parse_resume_pdf(file: UploadFile = File(...)):
    """Basic PDF parsing - returns full text"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    content = await file.read()
    text = parse_pdf(content)
    cleaned = clean_text(text)
    
    return {
        "text": cleaned,
        "length": len(cleaned)
    }


@app.post("/parse-resume-enhanced")
async def parse_resume_enhanced(file: UploadFile = File(...)):
    """
    Enhanced PDF parsing - extracts both full text and technical skills section
    Returns detailed extraction metadata
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    content = await file.read()
    result = parse_pdf_with_skills_extraction(content)
    
    return {
        "full_text": result["full_text"],
        "skills_section": result["skills_section"],
        "extraction_method": result["extraction_method"],
        "full_text_length": len(result["full_text"]),
        "skills_section_length": result["skills_section_length"],
        "extraction_successful": result["skills_section_length"] > 20
    }


@app.post("/parse-resume-text")
def parse_resume_text(data: ResumeText):
    """Parse text resume"""
    cleaned = clean_text(data.text)
    return {
        "text": cleaned,
        "length": len(cleaned)
    }


@app.post("/extract-skills")
def extract_skills_endpoint(data: ResumeText):
    """Extract skills from text"""
    skills = extract_skills(data.text)
    return {
        "skills": skills,
        "count": len(skills)
    }


@app.post("/analyze-gaps")
def analyze_gaps_endpoint(data: AnalysisRequest):
    """Analyze skill gaps"""
    gap_analysis = analyze_skill_gaps(data.skills, data.role)
    return gap_analysis


@app.post("/recommend-courses")
def recommend_courses_endpoint(data: AnalysisRequest):
    """Recommend courses based on gaps"""
    gap_analysis = analyze_skill_gaps(data.skills, data.role)
    recommendations = recommend_courses(gap_analysis)
    return {
        "recommendations": recommendations,
        "count": len(recommendations)
    }


@app.get("/available-roles")
def get_available_roles():
    """Get all available job roles"""
    return {
        "roles": list(JOB_ROLES.keys()),
        "count": len(JOB_ROLES)
    }


@app.get("/skills-taxonomy")
def get_skills_taxonomy():
    """Get complete skills taxonomy"""
    return {
        "taxonomy": SKILLS_TAXONOMY,
        "total_skills": len(ALL_SKILLS)
    }


@app.post("/analyze-resume")
async def analyze_resume_complete(
    file: UploadFile = File(...),
    role: str = Form("Data Scientist"),
    use_enhanced_extraction: bool = Form(True)
):
    """
    Complete analysis workflow with enhanced extraction:
    1. Parse PDF resume (with intelligent skills section extraction)
    2. Extract skills (from skills section primarily)
    3. Analyze gaps
    4. Recommend courses
    5. Generate AI counselor advice and roadmap
    """
    try:
        # Step 1: Enhanced PDF parsing
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files supported")
        
        content = await file.read()
        
        if use_enhanced_extraction:
            # Use enhanced extraction with skills section focus
            parsed_result = parse_pdf_with_skills_extraction(content)
            
            # If valid skills section exists, use ONLY skills from that section
            if parsed_result["skills_section_length"] > 20:
                user_skills = extract_skills(parsed_result["skills_section"], use_skills_section_only=True)
                extraction_info = {
                    "method": parsed_result["extraction_method"],
                    "skills_section_found": True,
                    "skills_extracted_from": "section_only",
                    "total_unique_skills": len(user_skills)
                }
            else:
                # Fallback: extract from full text if no valid skills section found
                user_skills = extract_skills(parsed_result["full_text"], use_skills_section_only=False)
                extraction_info = {
                    "method": parsed_result["extraction_method"],
                    "skills_section_found": False,
                    "skills_extracted_from": "full_text_fallback",
                    "total_unique_skills": len(user_skills)
                }
        else:
            # Use basic extraction
            resume_text = parse_pdf(content)
            cleaned_text = clean_text(resume_text)
            user_skills = extract_skills(cleaned_text)
            extraction_info = {
                "method": "basic_full_text_extraction",
                "total_skills": len(user_skills)
            }
        
        # Step 2: Analyze gaps
        gap_analysis = analyze_skill_gaps(user_skills, role)
        
        # Step 3: Recommend courses
        recommendations = recommend_courses(gap_analysis)
        
        # Step 4: Generate AI counselor advice
        ai_advice = generate_ai_counselor_advice(gap_analysis, user_skills, recommendations)
        
        return {
            "success": True,
            "extraction_info": extraction_info,
            "user_skills": sorted(user_skills),
            "gap_analysis": gap_analysis,
            "recommendations": recommendations,
            "ai_counselor": ai_advice["counseling"],
            "roadmap": ai_advice["roadmap"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "all-MiniLM-L6-v2",
        "version": "2.0-enhanced",
        "features": ["intelligent_skills_extraction", "multi_strategy_parsing", "ai_assisted_detection"]
    }


