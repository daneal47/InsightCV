# Developed by dnoobnerd [https://dnoobnerd.netlify.app]     Made with Streamlit

import os
import sys
import io
import re
import time
import datetime
import random
import socket
import secrets
import platform
import base64

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import nltk
import spacy
try:
    spacy.load('en_core_web_sm')
except Exception:
    # NEW: On Streamlit Cloud, runtime pip installs are blocked (Permission
    # denied), so this download only ever succeeds locally. The model is
    # now installed at BUILD time via requirements.txt instead — this stays
    # only as a harmless local-dev fallback and must never crash the app.
    try:
        from spacy.cli import download
        download('en_core_web_sm')
    except Exception as _model_dl_err:
        print(f"[startup] Could not auto-download en_core_web_sm (expected on some hosts): {_model_dl_err}")

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# ============================================================================
# NEW: In-memory replacement for pyresparser's ResumeParser class.
#
# The original fix was applied by physically replacing
# venvapp/Lib/site-packages/pyresparser/resume_parser.py on disk. On
# Streamlit Cloud, the app process does NOT have write permission to its own
# site-packages directory at runtime (confirmed by an earlier
# "Permission denied" error), so writing a patched file there silently fails
# and the original, unguarded pyresparser code keeps running instead.
#
# The fix below sidesteps the filesystem entirely: it imports only
# `pyresparser.utils` (unmodified, safe) and defines our own corrected
# ResumeParser class directly here, purely in memory. This guarantees the
# fix is always active, regardless of any host's file permissions.
# ============================================================================
from pyresparser import utils as _pyresparser_utils
from spacy.matcher import Matcher as _Matcher


def _extract_name_spacy3(nlp_text, matcher):
    """Re-implementation of pyresparser.utils.extract_name compatible with
    spaCy 3.x's Matcher.add() signature (patterns must be passed as a single
    list argument, not unpacked as *args like in spaCy 2.x)."""
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME', [pattern])
    matches = matcher(nlp_text)
    for _match_id, start, end in matches:
        span = nlp_text[start:end]
        if 'name' not in span.text.lower():
            return span.text
    return None


class ResumeParser(object):
    def __init__(self, resume, skills_file=None, custom_regex=None):
        nlp = spacy.load('en_core_web_sm')
        try:
            import pyresparser as _pyresparser_pkg
            custom_nlp = spacy.load(os.path.dirname(os.path.abspath(_pyresparser_pkg.__file__)))
        except Exception:
            # pyresparser's bundled custom NER model was trained for an old
            # spaCy 2.x format and can't load under spaCy 3.x — fall back to
            # the standard model instead of crashing. Name/Degree extraction
            # simply uses the standard NLP pipeline in this case.
            custom_nlp = nlp

        self.__skills_file = skills_file
        self.__custom_regex = custom_regex
        self.__matcher = _Matcher(nlp.vocab)
        self.__details = {
            'name': None,
            'email': None,
            'mobile_number': None,
            'skills': None,
            'degree': None,
            'no_of_pages': None,
        }
        self.__resume = resume
        if not isinstance(self.__resume, io.BytesIO):
            ext = os.path.splitext(self.__resume)[1].split('.')[1]
        else:
            ext = self.__resume.name.split('.')[1]
        self.__text_raw = _pyresparser_utils.extract_text(self.__resume, '.' + ext)
        self.__text = ' '.join(self.__text_raw.split())
        self.__nlp = nlp(self.__text)
        self.__custom_nlp = custom_nlp(self.__text_raw)
        self.__noun_chunks = list(self.__nlp.noun_chunks)
        self.__get_basic_details()

    def get_extracted_data(self):
        return self.__details

    def __get_basic_details(self):
        cust_ent = _pyresparser_utils.extract_entities_wih_custom_model(self.__custom_nlp)
        name = _extract_name_spacy3(self.__nlp, matcher=self.__matcher)
        email = _pyresparser_utils.extract_email(self.__text)
        mobile = _pyresparser_utils.extract_mobile_number(self.__text, self.__custom_regex)
        skills = _pyresparser_utils.extract_skills(self.__nlp, self.__noun_chunks, self.__skills_file)

        try:
            self.__details['name'] = cust_ent['Name'][0]
        except (IndexError, KeyError):
            self.__details['name'] = name

        self.__details['email'] = email
        self.__details['mobile_number'] = mobile
        self.__details['skills'] = skills
        self.__details['no_of_pages'] = _pyresparser_utils.get_number_of_pages(self.__resume)

        try:
            self.__details['degree'] = cust_ent['Degree']
        except KeyError:
            pass

from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter


import streamlit as st
import pandas as pd
import pymysql
import geocoder
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from streamlit_tags import st_tags
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Courses import (
    ds_course, web_course, android_course, ios_course, 
    uiux_course, cyber_course, devops_course, data_analyst_course, 
    marketing_course, game_course, resume_videos, interview_videos
)


###### Preprocessing functions ######


# Generates a link allowing the data in a given panda dataframe to be downloaded in csv format 
def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    ## bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()      
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Reads Pdf file and check_extractable
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    ## close open handles
    converter.close()
    fake_file_handle.close()
    return text


# show uploaded file path to view pdf_display
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    # NEW: Render each PDF page as an image instead of embedding via
    # <iframe src="data:application/pdf;base64,...">. That approach worked
    # on localhost but is blocked by Chrome's stricter Content-Security-Policy
    # on Streamlit Cloud. Rendering as images has no such restriction and
    # works identically on every browser and device.
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap(dpi=120)
            st.image(pix.tobytes("png"), use_column_width=True)
        doc.close()
    except Exception as e:
        st.warning(f"Could not render PDF preview ({e}). Use the download button below to view the file.")

    st.download_button(
        label="📥 Download / View Resume PDF",
        data=pdf_bytes,
        file_name=os.path.basename(file_path),
        mime="application/pdf"
    )


# course recommendations which has data already loaded from Courses.py
def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course = []
    ## slider to choose from range 1-10
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


# NEW: computes semantic similarity between the resume text and a job description
# using TF-IDF vectorization + Cosine Similarity (covers "المقارنة الدلالية"
# requested in the project proposal, beyond simple keyword matching)
def semantic_match_score(resume_text, job_description_text):
    documents = [resume_text, job_description_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0] * 100, 2)


# NEW: module-level field keyword map + helper functions, used by the
# University batch-analysis section (kept separate from the detailed
# User-section logic so the original, already-tested flow isn't touched)
FIELD_KEYWORDS = {
    'Data Science': ['tensorflow','keras','pytorch','machine learning','deep learning','flask','streamlit'],
    'Web Development': ['react','django','node js','react js','php','laravel','magento','wordpress','javascript','angular js','c#','asp.net','flask'],
    'Android Development': ['android','android development','flutter','kotlin','xml','kivy'],
    'iOS Development': ['ios','ios development','swift','cocoa','cocoa touch','xcode'],
    'UI-UX Development': ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience'],
    'Cybersecurity': ['cybersecurity','cyber security','penetration testing','ethical hacking','network security','firewall','siem','soc analyst','vulnerability assessment','kali linux','nmap','wireshark','metasploit','incident response','malware analysis','owasp'],
    'DevOps / Cloud Computing': ['devops','docker','kubernetes','ci/cd','jenkins','terraform','ansible','aws','azure','gcp','cloud computing','linux administration','helm','github actions','infrastructure as code'],
    'Data Analysis / BI': ['excel','power bi','tableau','data visualization','business intelligence','dashboard','data analyst','google analytics','looker','data analysis','pivot table'],
    'Digital Marketing': ['seo','sem','google ads','social media marketing','content marketing','digital marketing','email marketing','marketing analytics','facebook ads','hubspot','ppc','copywriting'],
    'Game Development': ['unity','unreal engine','game development','3d modeling','blender','game design','shader programming','c++','game developer'],
}


def predict_field_from_skills(skills):
    """Returns the first matching predicted field for a list of extracted skills."""
    for skill in skills:
        s = str(skill).lower()
        for field, keywords in FIELD_KEYWORDS.items():
            if s in keywords:
                return field
    return 'NA'


def predict_level(resume_text, no_of_pages):
    """Lightweight experience-level classifier, mirrors the logic used in the User section."""
    if not no_of_pages or no_of_pages < 1:
        return 'NA'
    text_lower = resume_text.lower()
    if 'internship' in text_lower:
        return 'Intermediate'
    elif 'experience' in text_lower:
        return 'Experienced'
    else:
        return 'Fresher'


def quick_resume_score(resume_text, skills):
    """Condensed version of the full Overall Score logic, used for fast batch analysis."""
    score = 0
    section_weights = {
        'objective': 5, 'education': 10, 'experience': 12, 'internship': 5,
        'skill': 5, 'hobbies': 3, 'interest': 3, 'achievement': 7,
        'certification': 7, 'project': 8
    }
    text_lower = resume_text.lower()
    for keyword, pts in section_weights.items():
        if keyword in text_lower:
            score += pts
    skills_score = min(round(len(skills) * 1.5), 15)
    score += skills_score
    project_mentions = len(re.findall(r'project', resume_text, re.IGNORECASE))
    projects_score = min(project_mentions * 5, 15)
    score += projects_score
    return min(score, 100)


###### Database Stuffs ######


# sql connector
connection = pymysql.connect(
    host='mysql-246371e5-danealaled47-1a77.l.aivencloud.com',
    user='avnadmin',
    password='AVNS_iAW_6QlTg-rv4Z2Z0lx',  # اضغط على أيقونة العين 👁️ بجانب Password لنسخها
    db='defaultdb',
    port=15214,
    ssl={'ca': 'App/ca.pem'}  # مطلوب لأن Aiven يتطلب اتصال آمن (SSL mode: REQUIRED)
)
cursor = connection.cursor()


# inserting miscellaneous data, fetched results, prediction and recommendation into user_data table
def insert_data(sec_token,ip_add,host_name,dev_user,os_name_ver,latlong,city,state,country,act_name,act_mail,act_mob,name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses,pdf_name):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (str(sec_token),str(ip_add),host_name,dev_user,os_name_ver,str(latlong),city,state,country,act_name,act_mail,act_mob,name,email,str(res_score),timestamp,str(no_of_pages),reco_field,cand_level,skills,recommended_skills,courses,pdf_name)
    cursor.execute(insert_sql, rec_values)
    connection.commit()


# inserting feedback data into user_feedback table
def insertf_data(feed_name,feed_email,feed_score,comments,Timestamp):
    DBf_table_name = 'user_feedback'
    insertfeed_sql = "insert into " + DBf_table_name + """
    values (0,%s,%s,%s,%s,%s)"""
    rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
    cursor.execute(insertfeed_sql, rec_values)
    connection.commit()


###### NEW: Theme (Dark/Light) — design layer only ######


TRANSLATIONS = {
    'sidebar_title': '### Menu',
    'choose_label': 'Choose a section:',
    'nav_User': 'Analyze Resume',
    'nav_Feedback': 'Feedback',
    'nav_About': 'About',
    'nav_University': 'University',
    'nav_Admin': 'Admin',
    'footer': 'InsightCV — An AI-Powered Resume Analyzer and Matcher using NLP',
    'upload_prompt': 'Upload Your Resume, And Get Smart Recommendations',
    'choose_resume': 'Choose your Resume',
    'basic_info': '**Your Basic info 👀**',
    'resume_analysis': '**Resume Analysis 🤘**',
    'skills_reco': '**Skills Recommendation 💡**',
    'resume_tips': '**Resume Tips & Ideas 🥂**',
    'resume_score': '**Resume Score 📝**',
    'jd_prompt': 'Paste a Job Description to check your match score (optional):',
    'feedback_form': 'Feedback form',
    'name': 'Name',
    'email': 'Email',
    'rate_us': 'Rate Us From 1 - 5',
    'comments': 'Comments',
    'submit': 'Submit',
    'admin_welcome': 'Welcome to Admin Side',
    'username': 'Username',
    'password': 'Password',
    'login': 'Login',
    'about_title': '**About The Tool - InsightCV**',
    'about_body': """
    <p align='justify'>
        A tool which parses information from a resume using natural language processing and finds the keywords, clusters them into sectors based on their keywords. And lastly shows recommendations, predictions, and analytics to the applicant based on keyword matching and semantic job matching.
    </p>
    <p align="justify">
        <b>How to use it: -</b> <br/><br/>
        <b>Analyze Resume -</b> <br/>
        Fill the required fields and upload your resume in PDF format. Just sit back and relax, our tool will do the magic on its own.<br/><br/>
        <b>Feedback -</b> <br/>
        A place where you can suggest feedback about the tool.<br/><br/>
        <b>Admin -</b> <br/>
        For login use <b>admin</b> as username and <b>admin@resume-analyzer</b> as password. It will load all the required data and analysis.
    </p>
    """,
}


def apply_theme_and_language():
    """Injects a full professional dark-mode design system: typography and a
    cohesive color palette applied consistently across every Streamlit
    component (sidebar, buttons, inputs, tables, alerts, progress bar, tabs,
    expanders, etc). It also neutralises the many hardcoded inline hex colors
    scattered throughout this file (e.g. style='color:#021659') so they
    resolve to theme-aware colors instead of breaking against a dark background."""

    direction = "ltr"
    align_start = "left"
    align_end = "right"

    # NEW: Dark mode is now the only theme — the light-mode branch and the
    # toggle button were removed per request.
    bg            = "#0b0e14"
    surface       = "#141a24"
    surface_alt   = "#1b2230"
    border        = "#2a3142"
    text_primary  = "#e9edf5"
    text_secondary= "#a9b1c6"
    text_muted    = "#7c8496"
    heading       = "#8fb0ff"
    accent        = "#4f7cff"
    accent_hover  = "#6f92ff"
    accent_text   = "#0b0e14"
    success       = "#2fd47a"
    danger        = "#ff6b85"
    warning       = "#ffb26a"
    shadow        = "0 8px 24px rgba(0,0,0,0.45)"

    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

            :root, .stApp {{
                /* --- our own design tokens --- */
                --bg: {bg};
                --surface: {surface};
                --surface-alt: {surface_alt};
                --border: {border};
                --text-primary: {text_primary};
                --text-secondary: {text_secondary};
                --text-muted: {text_muted};
                --heading: {heading};
                --accent: {accent};
                --accent-hover: {accent_hover};
                --accent-text: {accent_text};
                --success: {success};
                --danger: {danger};
                --warning: {warning};
                --shadow: {shadow};
                --font-main: 'Cairo', 'Inter', -apple-system, sans-serif;
                --radius: 14px;

                /* --- Streamlit's OWN theme variables ---
                   Native/BaseWeb widgets (select dropdowns, menus, sliders,
                   the file-uploader) read THESE variables directly, and some
                   of them render in a portal appended to <body>, outside
                   .stApp. Without overriding these too, those widgets keep
                   using Streamlit's default theme regardless of our custom
                   colors above - producing white-on-white / black-on-black
                   text. Setting them on :root makes portaled elements pick
                   them up as well. */
                --primary-color: {accent};
                --background-color: {bg};
                --secondary-background-color: {surface};
                --text-color: {text_primary};
                --font: 'Cairo', 'Inter', -apple-system, sans-serif;
            }}

            /* Force Streamlit's color-scheme so native form controls
               (checkboxes, scrollbars, etc.) render correctly too. */
            html {{ color-scheme: dark; }}

            html, body, .stApp {{
                direction: {direction};
                background-color: var(--bg) !important;
                color: var(--text-primary) !important;
                font-family: var(--font-main) !important;
            }}

            * {{ font-family: var(--font-main) !important; }}
            /* Font Awesome icons are glyphs from their OWN icon font - the
               blanket rule above was overwriting that font and silently
               turning every icon (theme/language toggles) into an empty
               box. Restore their real font explicitly. */
            i.fa-solid, i.fa-regular, i[class*="fa-"] {{
                font-family: "Font Awesome 6 Free", "FontAwesome" !important;
                font-weight: 900 !important;
            }}

            /* NEW: Streamlit's own built-in icons (sidebar collapse/expand
               arrow, expander chevrons, etc.) are rendered as ligature text
               using the Material Symbols icon font. The blanket "*" rule
               above was overriding that font too, so instead of an arrow
               icon it displayed literal text like "keyboard_double_arrow_right".
               Restore the icon font for every pattern Streamlit uses. */
            [data-testid="stIconMaterial"],
            span[class*="material-symbols"],
            span[class*="material-icons"],
            [class*="material-symbols-outlined"],
            [class*="material-symbols-rounded"] {{
                font-family: "Material Symbols Outlined", "Material Symbols Rounded", "Material Icons" !important;
                -webkit-font-feature-settings: "liga";
                font-feature-settings: "liga";
            }}

            /* ---------- Typography ---------- */
            h1, h2, h3, h4, h5, h6 {{
                color: var(--heading) !important;
                font-weight: 700 !important;
                text-align: {align_start} !important;
                letter-spacing: 0;
            }}
            p, label, span, li, div, .stMarkdown {{
                text-align: {align_start} !important;
                color: var(--text-primary) !important;
            }}
            a {{ color: var(--accent) !important; font-weight: 600; }}

            /* ---------- Sidebar ---------- */
            section[data-testid="stSidebar"] {{
                background-color: var(--surface) !important;
                border-{align_end}: 1px solid var(--border);
                direction: {direction};
            }}
            section[data-testid="stSidebar"] * {{ text-align: {align_start} !important; }}

            /* ---------- Buttons ---------- */
            .stButton>button, .stDownloadButton>button {{
                background-color: var(--accent) !important;
                color: var(--accent-text) !important;
                border: 1px solid var(--accent) !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                padding: 0.5rem 1.4rem !important;
                transition: all 0.2s ease-in-out;
                box-shadow: var(--shadow);
            }}
            .stButton>button:hover, .stDownloadButton>button:hover {{
                background-color: var(--accent-hover) !important;
                border-color: var(--accent-hover) !important;
                transform: translateY(-1px);
            }}

            /* ---------- Inputs / Select / Sliders ---------- */
            .stTextInput input, .stTextArea textarea, .stNumberInput input,
            div[data-baseweb="select"] > div, div[data-baseweb="input"] {{
                background-color: var(--surface) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--border) !important;
                border-radius: 10px !important;
                direction: {direction};
                text-align: {align_start} !important;
            }}
            div[data-baseweb="select"] * {{ color: var(--text-primary) !important; }}
            .stSlider [data-baseweb="slider"] > div > div {{ background: var(--accent) !important; }}

            /* ---------- Cards / Containers / Expanders ---------- */
            div[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {{
                background-color: var(--surface) !important;
                border: 1px solid var(--border) !important;
                border-radius: var(--radius) !important;
                box-shadow: var(--shadow);
            }}

            /* ---------- Alerts ---------- */
            div[data-testid="stAlert"] {{
                border-radius: 10px !important;
                direction: {direction};
                text-align: {align_start} !important;
            }}
            div[data-baseweb="notification"] {{ border-radius: 10px !important; }}

            /* ---------- Progress bar ---------- */
            .stProgress > div > div > div > div {{
                background-color: var(--accent) !important;
                border-radius: 8px !important;
            }}
            .stProgress > div > div > div {{
                background-color: var(--surface-alt) !important;
                border-radius: 8px !important;
            }}

            /* ---------- Tabs ---------- */
            button[data-baseweb="tab"] {{ color: var(--text-secondary) !important; font-weight: 600; }}
            button[aria-selected="true"][data-baseweb="tab"] {{ color: var(--accent) !important; }}
            div[data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; }}

            /* ---------- Dataframes / Tables ---------- */
            div[data-testid="stDataFrame"] {{
                border: 1px solid var(--border) !important;
                border-radius: 10px !important;
                overflow: hidden;
            }}

            /* ---------- Data-viz components: force internal LTR ----------
               Streamlit's dataframe grid (canvas-based) and Plotly charts
               (SVG-based) calculate their internal layout assuming LTR.
               Inheriting our page-level RTL direction breaks their position
               math - column headers get clipped to a single letter, and
               Plotly charts can render entirely blank. These components
               show tabular/numeric data anyway (not reading-direction
               sensitive), so they're pinned to LTR internally no matter
               which language is active; only their outer placement on the
               page still follows the page's RTL flow normally. */
            [data-testid="stDataFrame"], [data-testid="stDataFrame"] *,
            [data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] * {{
                direction: ltr !important;
                text-align: left !important;
            }}

            /* ---------- File uploader ----------
               NOTE: Streamlit renders the dropzone as a <section>, not a
               <div> - the previous div-qualified selector never matched,
               which is why this box stayed on Streamlit's default dark
               styling even in light mode. Using a plain attribute selector
               (no tag requirement) fixes that. */
            [data-testid="stFileUploader"],
            [data-testid="stFileUploaderDropzone"] {{
                background-color: var(--surface-alt) !important;
                border: 1.5px dashed var(--border) !important;
                border-radius: var(--radius) !important;
            }}
            [data-testid="stFileUploaderDropzone"] * {{ color: var(--text-secondary) !important; }}
            [data-testid="stFileUploaderDropzone"] button {{
                background-color: var(--accent) !important;
                border: 1px solid var(--accent) !important;
                border-radius: 8px !important;
            }}
            [data-testid="stFileUploaderDropzone"] button * {{ color: var(--accent-text) !important; }}


            /* ---------- Widget labels & captions (the text ABOVE
               inputs/selects - was rendering invisible before) ---------- */
            [data-testid="stWidgetLabel"],
            [data-testid="stWidgetLabel"] *,
            [data-testid="stCaptionContainer"],
            [data-testid="stCaptionContainer"] * {{
                color: var(--text-primary) !important;
                opacity: 1 !important;
            }}

            /* ---------- Selectbox closed-state value text ---------- */
            div[data-baseweb="select"] div {{ color: var(--text-primary) !important; }}

            /* ---------- Selectbox / dropdown OPEN menu ----------
               This list is a BaseWeb "popover" portaled to <body>, outside
               .stApp, so it must be targeted directly rather than relying
               on inheritance. */
            div[data-baseweb="popover"],
            ul[data-baseweb="menu"] {{
                background-color: var(--surface) !important;
                border: 1px solid var(--border) !important;
                border-radius: 10px !important;
                box-shadow: var(--shadow) !important;
            }}
            ul[data-baseweb="menu"] li,
            li[data-baseweb="menu-item"],
            li[role="option"] {{
                background-color: var(--surface) !important;
                color: var(--text-primary) !important;
            }}
            ul[data-baseweb="menu"] li:hover,
            li[data-baseweb="menu-item"]:hover,
            li[aria-selected="true"] {{
                background-color: var(--surface-alt) !important;
                color: var(--accent) !important;
            }}

            /* ---------- Metric ---------- */
            div[data-testid="stMetric"] {{
                background-color: var(--surface) !important;
                border: 1px solid var(--border) !important;
                border-radius: var(--radius) !important;
                padding: 0.75rem 1rem !important;
                box-shadow: var(--shadow);
            }}

            /* ---------- Neutralise legacy hardcoded inline colors ----------
               The rest of this app sets ad-hoc inline colors (e.g.
               style='color:#021659') that were only ever tuned for light
               mode. We re-map each of those known hex values to the correct
               theme token, with !important, so both dark and light mode
               stay legible without touching every call site. */
            [style*="color: #021659"], [style*="color:#021659"] {{ color: var(--heading) !important; }}
            [style*="color: #092851"], [style*="color:#092851"] {{ color: var(--text-secondary) !important; }}
            [style*="color: #000000"], [style*="color:#000000"] {{ color: var(--text-primary) !important; }}
            [style*="color: #1ed760"], [style*="color:#1ed760"],
            [style*="color: #1DB954"], [style*="color:#1DB954"] {{ color: var(--success) !important; }}
            [style*="color: #d73b5c"], [style*="color:#d73b5c"] {{ color: var(--danger) !important; }}
            [style*="color: #fba171"], [style*="color:#fba171"] {{ color: var(--warning) !important; }}

            footer {{ visibility: hidden; }}

            /* ---------- Scrollbar ----------
               Default browser scrollbar can be invisible against a light
               background depending on OS theme; style it explicitly so it
               stays visible and on-brand in both modes. */
            ::-webkit-scrollbar {{ width: 12px; height: 12px; }}
            ::-webkit-scrollbar-track {{ background: var(--surface-alt); }}
            ::-webkit-scrollbar-thumb {{
                background-color: var(--text-muted);
                border-radius: 8px;
                border: 3px solid var(--surface-alt);
            }}
            ::-webkit-scrollbar-thumb:hover {{ background-color: var(--text-secondary); }}
            * {{ scrollbar-width: thin; scrollbar-color: var(--text-muted) var(--surface-alt); }}

            /* ---------- Neutralise hardcoded "text-align: left" ----------
               Dozens of st.markdown() calls throughout this app hardcode
               style='text-align: left' - fine for English, but it silently
               overrides our RTL layout whenever Arabic is selected (these
               inline styles win over a plain tag-selector rule). Re-mapping
               them to the current language's direction here fixes every one
               of those call sites at once, without editing each of them. */
            [style*="text-align: left"], [style*="text-align:left"] {{
                text-align: {align_start} !important;
            }}
        </style>
    """, unsafe_allow_html=True)


###### Setting Page Configuration (favicon, Logo, Title) ######


st.set_page_config(
   page_title="AI Resume Analyzer",
   page_icon='./Logo/recommend.png',
)


###### Main function run() ######


def run():

    # NEW: Dark mode is now the only theme (toggle button removed per request)
    t = TRANSLATIONS
    apply_theme_and_language()

    # (Logo, Heading, Sidebar etc)
    img = Image.open('./App/Logo/RESUM.jpg')
    st.image(img)
    st.sidebar.markdown(t['sidebar_title'])
    activities = ["User", "Feedback", "About", "University", "Admin"]
    choice = st.sidebar.selectbox(t['choose_label'], activities, format_func=lambda x: t['nav_' + x])

    st.sidebar.markdown('---')
    st.sidebar.markdown(f"<p style='text-align:center; font-size: 0.85em; opacity: 0.75;'>{t['footer']}</p>", unsafe_allow_html=True)

    ###### Creating Database and Table ######


    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)


    # Create table user_data and user_feedback
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                    sec_token varchar(20) NOT NULL,
                    ip_add varchar(50) NULL,
                    host_name varchar(50) NULL,
                    dev_user varchar(50) NULL,
                    os_name_ver varchar(50) NULL,
                    latlong varchar(50) NULL,
                    city varchar(50) NULL,
                    state varchar(50) NULL,
                    country varchar(50) NULL,
                    act_name varchar(50) NOT NULL,
                    act_mail varchar(50) NOT NULL,
                    act_mob varchar(20) NOT NULL,
                    Name varchar(500) NOT NULL,
                    Email_ID VARCHAR(500) NOT NULL,
                    resume_score VARCHAR(8) NOT NULL,
                    Timestamp VARCHAR(50) NOT NULL,
                    Page_no VARCHAR(5) NOT NULL,
                    Predicted_Field BLOB NOT NULL,
                    User_level BLOB NOT NULL,
                    Actual_skills BLOB NOT NULL,
                    Recommended_skills BLOB NOT NULL,
                    Recommended_courses BLOB NOT NULL,
                    pdf_name varchar(50) NOT NULL,
                    PRIMARY KEY (ID)
                    );
                """
    cursor.execute(table_sql)


    DBf_table_name = 'user_feedback'
    tablef_sql = "CREATE TABLE IF NOT EXISTS " + DBf_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                        feed_name varchar(50) NOT NULL,
                        feed_email VARCHAR(50) NOT NULL,
                        feed_score VARCHAR(5) NOT NULL,
                        comments VARCHAR(100) NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        PRIMARY KEY (ID)
                    );
                """
    cursor.execute(tablef_sql)


    ###### CODE FOR CLIENT SIDE (USER) ######

    if choice == 'User':
        
        # Collecting Miscellaneous Information
        act_name = st.text_input('Name*')
        act_mail = st.text_input('Mail*')
        act_mob  = st.text_input('Mobile Number*')
        sec_token = secrets.token_urlsafe(12)
        try:
            host_name = socket.gethostname()
            ip_add = socket.gethostbyname(host_name)
        except Exception:
            host_name = "StreamlitServer"
            ip_add = "127.0.0.1"

        try:
           dev_user = os.getlogin()
        except Exception:
            dev_user = os.environ.get("USER", "default_user")

            os_name_ver = platform.system() + " " + platform.release()

        try:
            g = geocoder.ip('me')
            latlong = g.latlng if (g and g.latlng) else [0.0, 0.0]
        except Exception:
            latlong = [0.0, 0.0]
        ### Wrapped in try/except so a slow/unreachable geocoding service
        ### doesn't crash the whole app (this data is only used for admin stats)
        try:
            geolocator = Nominatim(user_agent="http", timeout=10)
            location = geolocator.reverse(latlong, language='en')
            address = location.raw['address']
            cityy = address.get('city', '')
            statee = address.get('state', '')
            countryy = address.get('country', '')
        except Exception:
            cityy = ''
            statee = ''
            countryy = ''
        city = cityy
        state = statee
        country = countryy


        # Upload Resume
        st.markdown(f"<h5 style='text-align: left; color: #021659;'>{t['upload_prompt']}</h5>", unsafe_allow_html=True)
        
        ## file upload in pdf format
        pdf_file = st.file_uploader(t['choose_resume'], type=["pdf"])
        if pdf_file is not None:
            with st.spinner('Hang On While We Cook Magic For You...'):
                time.sleep(4)
        
            ### saving the uploaded resume to folder
            target_dir = './Uploaded_Resumes'
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            
            save_image_path = os.path.join(target_dir, pdf_file.name)
            pdf_name = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            ### parsing and extracting whole resume 
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                
                ## Get the whole resume data into resume_text
                resume_text = pdf_reader(save_image_path)

                ## Showing Analyzed data from (resume_data)
                st.header(t['resume_analysis'])
                st.success("Hello "+ resume_data['name'])
                st.subheader(t['basic_info'])
                try:
                    st.text('Name: '+resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Degree: '+str(resume_data['degree']))                    
                    st.text('Resume pages: '+str(resume_data['no_of_pages']))

                except:
                    pass
                ## Predicting Candidate Experience Level 

                ### Trying with different possibilities
                cand_level = ''
                if resume_data['no_of_pages'] < 1:                
                    cand_level = "NA"
                    st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>''',unsafe_allow_html=True)
                
                #### if internship then intermediate level
                elif 'INTERNSHIP' in resume_text:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                elif 'INTERNSHIPS' in resume_text:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                elif 'Internship' in resume_text:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                elif 'Internships' in resume_text:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                
                #### if Work Experience/Experience then Experience level
                elif 'EXPERIENCE' in resume_text:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
                elif 'WORK EXPERIENCE' in resume_text:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
                elif 'Experience' in resume_text:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
                elif 'Work Experience' in resume_text:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
                else:
                    cand_level = "Fresher"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at Fresher level!!''',unsafe_allow_html=True)


                ## Skills Analyzing and Recommendation
                st.subheader(t['skills_reco'])
                
                ### Current Analyzed Skills
                keywords = st_tags(label='### Your Current Skills',
                text='See our skills recommendation below',value=resume_data['skills'],key = '1  ')

                ### NEW: Job Description input for Semantic Matching
                st.markdown(f"<h5 style='text-align: left; color: #021659;'>{t['jd_prompt']}</h5>", unsafe_allow_html=True)
                job_description_input = st.text_area(
                    "Job Description",
                    height=150,
                    placeholder="مثال: We are looking for a Python developer with experience in machine learning, SQL, and data analysis..."
                )

                ### Keywords for Recommendations
                ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit']
                web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress','javascript', 'angular js', 'C#', 'Asp.net', 'flask']
                android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
                ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
                uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience']
                # NEW: additional sector keyword lists (widening the "clustering by sector" logic)
                cyber_keyword = ['cybersecurity','cyber security','penetration testing','ethical hacking','network security','firewall','siem','soc analyst','vulnerability assessment','kali linux','nmap','wireshark','metasploit','incident response','malware analysis','owasp']
                devops_keyword = ['devops','docker','kubernetes','ci/cd','jenkins','terraform','ansible','aws','azure','gcp','cloud computing','linux administration','helm','github actions','infrastructure as code']
                data_analyst_keyword = ['excel','power bi','tableau','data visualization','business intelligence','dashboard','data analyst','google analytics','looker','data analysis','pivot table']
                marketing_keyword = ['seo','sem','google ads','social media marketing','content marketing','digital marketing','email marketing','marketing analytics','facebook ads','hubspot','ppc','copywriting']
                game_dev_keyword = ['unity','unreal engine','game development','3d modeling','blender','game design','shader programming','c++','game developer']
                n_any = ['english','communication','writing', 'microsoft office', 'leadership','customer management', 'social media']
                ### Skill Recommendations Starts                
                recommended_skills = []
                reco_field = ''
                rec_course = ''

                ### condition starts to check skills from keywords and predict field
                for i in resume_data['skills']:
                
                    #### Data science recommendation
                    if i.lower() in ds_keyword:
                        print(i.lower())
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '2')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(ds_course)
                        break

                    #### Web development recommendation
                    elif i.lower() in web_keyword:
                        print(i.lower())
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '3')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(web_course)
                        break

                    #### Android App Development
                    elif i.lower() in android_keyword:
                        print(i.lower())
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '4')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(android_course)
                        break

                    #### IOS App Development
                    elif i.lower() in ios_keyword:
                        print(i.lower())
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '5')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(ios_course)
                        break

                    #### Ui-UX Recommendation
                    elif i.lower() in uiux_keyword:
                        print(i.lower())
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '6')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = course_recommender(uiux_course)
                        break

                    #### NEW: Cybersecurity recommendation
                    elif i.lower() in cyber_keyword:
                        print(i.lower())
                        reco_field = 'Cybersecurity'
                        st.success("** Our analysis says you are looking for Cybersecurity Jobs **")
                        recommended_skills = ['Network Security','Penetration Testing','Ethical Hacking','SIEM','Incident Response','Vulnerability Assessment','Kali Linux','Wireshark','Metasploit','OWASP Top 10']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '7')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        rec_course = course_recommender(cyber_course)
                        break

                    #### NEW: DevOps / Cloud Computing recommendation
                    elif i.lower() in devops_keyword:
                        print(i.lower())
                        reco_field = 'DevOps / Cloud Computing'
                        st.success("** Our analysis says you are looking for DevOps / Cloud Computing Jobs **")
                        recommended_skills = ['Docker','Kubernetes','CI/CD','Jenkins','Terraform','Ansible','AWS','Azure','GCP','Infrastructure as Code','Linux Administration']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '8')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        rec_course = course_recommender(devops_course)
                        break

                    #### NEW: Data Analysis / Business Intelligence recommendation
                    elif i.lower() in data_analyst_keyword:
                        print(i.lower())
                        reco_field = 'Data Analysis / Business Intelligence'
                        st.success("** Our analysis says you are looking for Data Analysis / BI Jobs **")
                        recommended_skills = ['Power BI','Tableau','Advanced Excel','SQL','Data Visualization','Business Intelligence','Google Analytics','Dashboard Design','Statistics']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '9')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        rec_course = course_recommender(data_analyst_course)
                        break

                    #### NEW: Digital Marketing recommendation
                    elif i.lower() in marketing_keyword:
                        print(i.lower())
                        reco_field = 'Digital Marketing'
                        st.success("** Our analysis says you are looking for Digital Marketing Jobs **")
                        recommended_skills = ['SEO','SEM','Google Ads','Social Media Marketing','Content Marketing','Email Marketing','Marketing Analytics','Copywriting','HubSpot']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '10')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        rec_course = course_recommender(marketing_course)
                        break

                    #### NEW: Game Development recommendation
                    elif i.lower() in game_dev_keyword:
                        print(i.lower())
                        reco_field = 'Game Development'
                        st.success("** Our analysis says you are looking for Game Development Jobs **")
                        recommended_skills = ['Unity','Unreal Engine','C++','3D Modeling','Blender','Game Design','Shader Programming','Physics Engines']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '11')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                        rec_course = course_recommender(game_course)
                        break

                    #### For Not Any Recommendations
                    elif i.lower() in n_any:
                        print(i.lower())
                        reco_field = 'NA'
                        st.warning("** Currently our tool only predicts and recommends for Data Science, Web, Android, IOS and UI/UX Development**")
                        recommended_skills = ['No Recommendations']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Currently No Recommendations',value=recommended_skills,key = '6')
                        st.markdown('''<h5 style='text-align: left; color: #092851;'>Maybe Available in Future Updates</h5>''',unsafe_allow_html=True)
                        # course recommendation
                        rec_course = "Sorry! Not Available for this Field"
                        break

                ### NEW: Semantic Matching Result (TF-IDF + Cosine Similarity)
                if job_description_input.strip() != "":
                    match_percentage = semantic_match_score(resume_text, job_description_input)
                    st.subheader("**Job Match Score 🎯**")
                    st.markdown(f'''<h4 style='text-align: left; color: #1DB954;'>Semantic Match: {match_percentage}%</h4>''', unsafe_allow_html=True)
                    if match_percentage >= 60:
                        st.success("سيرتك الذاتية متوافقة بشكل جيد مع هذه الوظيفة!")
                    elif match_percentage >= 30:
                        st.warning("توافق متوسط — قد تحتاج لإبراز مهارات إضافية ذات صلة.")
                    else:
                        st.error("توافق منخفض — راجع الوصف الوظيفي وقارنه بمهاراتك الحالية.")
                else:
                    match_percentage = 0
                    st.info("💡 الصق وصف وظيفي أعلاه للحصول على درجة التوافق الدلالي مع سيرتك الذاتية.")


                ## Resume Scorer & Resume Writing Tips
                st.subheader(t['resume_tips'])
                resume_score = 0

                ### NEW: Scoring is split into two parts:
                ### 1) Section presence (65 pts total, rebalanced from the original 100 pts)
                ### 2) Content quality/quantity (35 pts) — skills count, projects count,
                ###    and a semantic job-match bonus (this addresses the roadmap item
                ###    "Add resume scoring criteria for skills and projects")

                ### Predicting Whether these key points are added to the resume
                if 'Objective' or 'Summary' in resume_text:
                    resume_score = resume_score+5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective/Summary</h4>''',unsafe_allow_html=True)                
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add your career objective, it will give your career intension to the Recruiters.</h4>''',unsafe_allow_html=True)

                if 'Education' or 'School' or 'College'  in resume_text:
                    resume_score = resume_score + 10
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Education Details</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Education. It will give Your Qualification level to the recruiter</h4>''',unsafe_allow_html=True)

                if 'EXPERIENCE' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Experience</h4>''',unsafe_allow_html=True)
                elif 'Experience' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Experience</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Experience. It will help you to stand out from crowd</h4>''',unsafe_allow_html=True)

                if 'INTERNSHIPS'  in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                elif 'INTERNSHIP'  in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                elif 'Internships'  in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                elif 'Internship'  in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Internships. It will help you to stand out from crowd</h4>''',unsafe_allow_html=True)

                if 'SKILLS'  in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                elif 'SKILL'  in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                elif 'Skills'  in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                elif 'Skill'  in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Skills. It will help you a lot</h4>''',unsafe_allow_html=True)

                if 'HOBBIES' in resume_text:
                    resume_score = resume_score + 3
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                elif 'Hobbies' in resume_text:
                    resume_score = resume_score + 3
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Hobbies. It will show your personality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',unsafe_allow_html=True)

                if 'INTERESTS'in resume_text:
                    resume_score = resume_score + 3
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Interest</h4>''',unsafe_allow_html=True)
                elif 'Interests'in resume_text:
                    resume_score = resume_score + 3
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Interest</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Interest. It will show your interest other that job.</h4>''',unsafe_allow_html=True)

                if 'ACHIEVEMENTS' in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
                elif 'Achievements' in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Achievements. It will show that you are capable for the required position.</h4>''',unsafe_allow_html=True)

                if 'CERTIFICATIONS' in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                elif 'Certifications' in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                elif 'Certification' in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Certifications. It will show that you have done some specialization for the required position.</h4>''',unsafe_allow_html=True)

                if 'PROJECTS' in resume_text:
                    resume_score = resume_score + 8
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                elif 'PROJECT' in resume_text:
                    resume_score = resume_score + 8
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                elif 'Projects' in resume_text:
                    resume_score = resume_score + 8
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                elif 'Project' in resume_text:
                    resume_score = resume_score + 8
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Projects. It will show that you have done work related the required position or not.</h4>''',unsafe_allow_html=True)

                ### NEW: Skills quantity score (up to 15 pts) — rewards a broader, more relevant skill set
                skills_count = len(resume_data.get('skills', []))
                skills_score = min(round(skills_count * 1.5), 15)
                resume_score = resume_score + skills_score
                if skills_count > 0:
                    st.markdown(f'''<h5 style='text-align: left; color: #1ed760;'>[+] You listed {skills_count} skills → +{skills_score} pts (Skills Quantity Score)</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] No skills detected. Add a clear Skills section with relevant keywords.</h4>''',unsafe_allow_html=True)

                ### NEW: Projects quantity score (up to 15 pts) — counts how many times "Project(s)" is
                ### mentioned in the resume as a rough proxy for the number of projects listed
                project_mentions = len(re.findall(r'project', resume_text, re.IGNORECASE))
                projects_score = min(project_mentions * 5, 15)
                resume_score = resume_score + projects_score
                if project_mentions > 0:
                    st.markdown(f'''<h5 style='text-align: left; color: #1ed760;'>[+] Detected {project_mentions} project mention(s) → +{projects_score} pts (Projects Quantity Score)</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] No projects detected. Listing real projects significantly boosts recruiter interest.</h4>''',unsafe_allow_html=True)

                ### NEW: Semantic Job Match bonus (up to 5 pts) — only applies if a job description was provided
                match_bonus = 0
                if job_description_input.strip() != "":
                    match_bonus = round((match_percentage / 100) * 5)
                    resume_score = resume_score + match_bonus
                    st.markdown(f'''<h5 style='text-align: left; color: #1ed760;'>[+] Semantic Job Match Bonus: +{match_bonus} pts (based on {match_percentage}% match)</h4>''',unsafe_allow_html=True)

                ### Final score is capped at 100 (bonus points can occasionally push slightly over)
                resume_score = min(resume_score, 100)

                st.subheader(t['resume_score'])
                
                ### Progress bar color now follows the theme (see apply_theme_and_language),
                ### no extra hardcoded override needed here.

                ### Score Bar
                my_bar = st.progress(0)
                score = 0
                for percent_complete in range(resume_score):
                    score +=1
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)

                ### Score
                st.success('** Your Resume Writing Score: ' + str(score)+'**')
                st.warning("** Note: This score combines section completeness (65 pts), skills & projects quantity (30 pts), and a job-match bonus (5 pts) when a job description is provided. **")

                # print(str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)


                ### Getting Current Date and Time
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)


                ## Calling insert_data to add all the data into user_data                
                insert_data(str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)

                ## Recommending Resume Writing Video
                st.header("**Bonus Video for Resume Writing Tips💡**")
                resume_vid = random.choice(resume_videos)
                st.video(resume_vid)

                ## Recommending Interview Preparation Video
                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = random.choice(interview_videos)
                st.video(interview_vid)

                ## On Successful Result 
                st.balloons()

            else:
                st.error('Something went wrong..')                


    ###### CODE FOR FEEDBACK SIDE ######
    elif choice == 'Feedback':   
        
        # timestamp 
        ts = time.time()
        cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        timestamp = str(cur_date+'_'+cur_time)

        # Feedback Form
        with st.form("my_form"):
            st.write(t['feedback_form'])            
            feed_name = st.text_input(t['name'])
            feed_email = st.text_input(t['email'])
            feed_score = st.slider(t['rate_us'], 1, 5)
            comments = st.text_input(t['comments'])
            Timestamp = timestamp        
            submitted = st.form_submit_button(t['submit'])
            if submitted:
                ## Calling insertf_data to add dat into user feedback
                insertf_data(feed_name,feed_email,feed_score,comments,Timestamp)    
                ## Success Message 
                st.success("Thanks! Your Feedback was recorded.") 
                ## On Successful Submit
                st.balloons()    


        # query to fetch data from user feedback table
        query = 'select * from user_feedback'        
        plotfeed_data = pd.read_sql(query, connection)                        


        # fetching feed_score from the query and getting the unique values and total value count 
        labels = plotfeed_data.feed_score.unique()
        values = plotfeed_data.feed_score.value_counts()


        # plotting pie chart for user ratings
        st.subheader("**Past User Rating's**")
        fig = px.pie(values=values, names=labels, title="Chart of User Rating Score From 1 - 5", color_discrete_sequence=px.colors.sequential.Aggrnyl)
        st.plotly_chart(fig)


        #  Fetching Comment History
        cursor.execute('select feed_name, comments from user_feedback')
        plfeed_cmt_data = cursor.fetchall()

        st.subheader("**User Comment's**")
        dff = pd.DataFrame(plfeed_cmt_data, columns=['User', 'Comment'])
        st.dataframe(dff, width=1000)

    
    ###### CODE FOR ABOUT PAGE ######
    elif choice == 'About':   

        st.subheader(t['about_title'])
        st.markdown(t['about_body'], unsafe_allow_html=True)


    ###### NEW: CODE FOR UNIVERSITY SIDE (BATCH ANALYSIS) ######
    elif choice == 'University':
        st.success('Welcome to the University Batch Analysis Panel 🎓')
        uni_intro = 'Upload multiple student resumes at once to get a batch-level readiness overview for the job market.'
        st.markdown(f"<h5 style='color: var(--heading);'>{uni_intro}</h5>", unsafe_allow_html=True)

        if 'uni_access_granted' not in st.session_state:
            st.session_state['uni_access_granted'] = False

        if not st.session_state['uni_access_granted']:
            uni_user = st.text_input("University Access Code", type='password', key='uni_pw')
            if st.button('Access Panel'):
                ## NOTE: placeholder access code — change this before real deployment
                if uni_user == 'university2026':
                    st.session_state['uni_access_granted'] = True
                    st.rerun()
                else:
                    st.error("Wrong Access Code Provided")
        else:
            uploaded_files = st.file_uploader("Upload Student Resumes (PDF, multiple files allowed)", type=["pdf"], accept_multiple_files=True)

            if uploaded_files:
                batch_results = []
                progress_text = st.empty()
                prog_bar = st.progress(0)

                for idx, uni_pdf_file in enumerate(uploaded_files):
                    progress_text.text(f"Analyzing {uni_pdf_file.name} ({idx+1}/{len(uploaded_files)})...")
                    uni_save_path = './Uploaded_Resumes/uni_' + uni_pdf_file.name
                    with open(uni_save_path, "wb") as f:
                        f.write(uni_pdf_file.getbuffer())

                    try:
                        uni_resume_data = ResumeParser(uni_save_path).get_extracted_data()
                        uni_resume_text = pdf_reader(uni_save_path)
                        if uni_resume_data:
                            uni_skills = uni_resume_data.get('skills', [])
                            uni_field = predict_field_from_skills(uni_skills)
                            uni_level = predict_level(uni_resume_text, uni_resume_data.get('no_of_pages', 0))
                            uni_score = quick_resume_score(uni_resume_text, uni_skills)
                            batch_results.append({
                                'File Name': uni_pdf_file.name,
                                'Name': uni_resume_data.get('name', 'N/A'),
                                'Predicted Field': uni_field,
                                'Experience Level': uni_level,
                                'Skills Count': len(uni_skills),
                                'Resume Score': uni_score
                            })
                        else:
                            batch_results.append({'File Name': uni_pdf_file.name, 'Name': 'Could not parse', 'Predicted Field': 'NA', 'Experience Level': 'NA', 'Skills Count': 0, 'Resume Score': 0})
                    except Exception:
                        batch_results.append({'File Name': uni_pdf_file.name, 'Name': 'Error while parsing', 'Predicted Field': 'NA', 'Experience Level': 'NA', 'Skills Count': 0, 'Resume Score': 0})

                    prog_bar.progress((idx + 1) / len(uploaded_files))

                progress_text.text("✅ Batch analysis complete!")
                df_batch = pd.DataFrame(batch_results)

                st.header("**Batch Analysis Results**")
                st.dataframe(df_batch, width=1000)
                st.markdown(get_csv_download_link(df_batch, 'University_Batch_Report.csv', 'Download Full Report'), unsafe_allow_html=True)

                ### Summary metrics
                st.subheader("**Batch Summary Statistics**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Students Analyzed", len(df_batch))
                col2.metric("Average Resume Score", round(df_batch['Resume Score'].mean(), 1))
                col3.metric("Average Skills Count", round(df_batch['Skills Count'].mean(), 1))

                ### Pie chart: predicted field distribution across the batch
                field_counts = df_batch['Predicted Field'].value_counts()
                fig_field = px.pie(values=field_counts.values, names=field_counts.index, title="Predicted Field Distribution Across Batch", color_discrete_sequence=px.colors.sequential.Aggrnyl)
                st.plotly_chart(fig_field)

                ### Pie chart: experience level distribution
                level_counts = df_batch['Experience Level'].value_counts()
                fig_level = px.pie(values=level_counts.values, names=level_counts.index, title="Experience Level Distribution", color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_level)

                ### Bar chart: job-market readiness overview
                st.subheader("**Job-Market Readiness Overview**")
                ready = len(df_batch[df_batch['Resume Score'] >= 70])
                moderate = len(df_batch[(df_batch['Resume Score'] >= 40) & (df_batch['Resume Score'] < 70)])
                needs_work = len(df_batch[df_batch['Resume Score'] < 40])
                readiness_df = pd.DataFrame({
                    'Readiness Level': ['Ready (Score ≥ 70)', 'Moderate (40-69)', 'Needs Improvement (<40)'],
                    'Count': [ready, moderate, needs_work]
                })
                fig_ready = px.bar(readiness_df, x='Readiness Level', y='Count', title="Batch Job-Market Readiness", color='Readiness Level')
                st.plotly_chart(fig_ready)

            if st.button('Logout'):
                st.session_state['uni_access_granted'] = False
                st.rerun()


    ###### CODE FOR ADMIN SIDE (ADMIN) ######
    else:
        st.success(t['admin_welcome'])

        # NEW: Persistent login state via session_state — without this, clicking
        # any button INSIDE the admin panel (like "Empty Database Now") triggers a
        # rerun where st.button(login) is False again, hiding the whole panel
        # before the nested action ever executes. This mirrors the same fix
        # already used in the University section.
        if 'admin_logged_in' not in st.session_state:
            st.session_state['admin_logged_in'] = False

        #  Admin Login
        ad_user = st.text_input(t['username'])
        ad_password = st.text_input(t['password'], type='password')

        login_clicked = st.button(t['login'])
        if login_clicked:
            if ad_user == 'admin' and ad_password == 'admin@resume-analyzer':
                st.session_state['admin_logged_in'] = True
            else:
                st.error("Wrong ID & Password Provided")

        if st.session_state['admin_logged_in']:
            if True:  # kept to preserve the original indentation of the block below
                
                ### Fetch miscellaneous data from user_data(table) and convert it into dataframe
                cursor.execute('''SELECT ID, ip_add, resume_score, convert(Predicted_Field using utf8), convert(User_level using utf8), city, state, country from user_data''')
                datanalys = cursor.fetchall()
                plot_data = pd.DataFrame(datanalys, columns=['Idt', 'IP_add', 'resume_score', 'Predicted_Field', 'User_Level', 'City', 'State', 'Country'])
                
                ### Total Users Count with a Welcome Message
                values = plot_data.Idt.count()
                st.success("Welcome ! Total %d " % values + " User's Have Used Our Tool : )")                
                
                ### Fetch user data from user_data(table) and convert it into dataframe
                cursor.execute('''SELECT ID, sec_token, ip_add, act_name, act_mail, act_mob, convert(Predicted_Field using utf8), Timestamp, Name, Email_ID, resume_score, Page_no, pdf_name, convert(User_level using utf8), convert(Actual_skills using utf8), convert(Recommended_skills using utf8), convert(Recommended_courses using utf8), city, state, country, latlong, os_name_ver, host_name, dev_user from user_data''')
                data = cursor.fetchall()                

                st.header("**User's Data**")
                df = pd.DataFrame(data, columns=['ID', 'Token', 'IP Address', 'Name', 'Mail', 'Mobile Number', 'Predicted Field', 'Timestamp',
                                                 'Predicted Name', 'Predicted Mail', 'Resume Score', 'Total Page',  'File Name',   
                                                 'User Level', 'Actual Skills', 'Recommended Skills', 'Recommended Course',
                                                 'City', 'State', 'Country', 'Lat Long', 'Server OS', 'Server Name', 'Server User',])
                
                ### Viewing the dataframe
                st.dataframe(df)
                
                ### Downloading Report of user_data in csv file
                st.markdown(get_csv_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)

                ### Fetch feedback data from user_feedback(table) and convert it into dataframe
                cursor.execute('''SELECT * from user_feedback''')
                data = cursor.fetchall()

                st.header("**User's Feedback Data**")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Feedback Score', 'Comments', 'Timestamp'])
                st.dataframe(df)

                ### query to fetch data from user_feedback(table)
                query = 'select * from user_feedback'
                plotfeed_data = pd.read_sql(query, connection)                        

                ### Analyzing All the Data's in pie charts

                # fetching feed_score from the query and getting the unique values and total value count 
                labels = plotfeed_data.feed_score.unique()
                values = plotfeed_data.feed_score.value_counts()
                
                # Pie chart for user ratings
                st.subheader("**User Rating's**")
                fig = px.pie(values=values, names=labels, title="Chart of User Rating Score From 1 - 5 🤗", color_discrete_sequence=px.colors.sequential.Aggrnyl)
                st.plotly_chart(fig)

                # fetching Predicted_Field from the query and getting the unique values and total value count                 
                labels = plot_data.Predicted_Field.unique()
                values = plot_data.Predicted_Field.value_counts()

                # Pie chart for predicted field recommendations
                st.subheader("**Pie-Chart for Predicted Field Recommendation**")
                fig = px.pie(df, values=values, names=labels, title='Predicted Field according to the Skills 👽', color_discrete_sequence=px.colors.sequential.Aggrnyl_r)
                st.plotly_chart(fig)

                # fetching User_Level from the query and getting the unique values and total value count                 
                labels = plot_data.User_Level.unique()
                values = plot_data.User_Level.value_counts()

                # Pie chart for User's👨‍💻 Experienced Level
                st.subheader("**Pie-Chart for User's Experienced Level**")
                fig = px.pie(df, values=values, names=labels, title="Pie-Chart 📈 for User's 👨‍💻 Experienced Level", color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig)

                # fetching resume_score from the query and getting the unique values and total value count                 
                labels = plot_data.resume_score.unique()                
                values = plot_data.resume_score.value_counts()

                # Pie chart for Resume Score
                st.subheader("**Pie-Chart for Resume Score**")
                fig = px.pie(df, values=values, names=labels, title='From 1 to 100 💯', color_discrete_sequence=px.colors.sequential.Agsunset)
                st.plotly_chart(fig)

                # fetching IP_add from the query and getting the unique values and total value count 
                labels = plot_data.IP_add.unique()
                values = plot_data.IP_add.value_counts()

                # Pie chart for Users
                st.subheader("**Pie-Chart for Users App Used Count**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based On IP Address 👥', color_discrete_sequence=px.colors.sequential.matter_r)
                st.plotly_chart(fig)

                # fetching City from the query and getting the unique values and total value count 
                labels = plot_data.City.unique()
                values = plot_data.City.value_counts()

                # Pie chart for City
                st.subheader("**Pie-Chart for City**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based On City 🌆', color_discrete_sequence=px.colors.sequential.Jet)
                st.plotly_chart(fig)

                # fetching State from the query and getting the unique values and total value count 
                labels = plot_data.State.unique()
                values = plot_data.State.value_counts()

                # Pie chart for State
                st.subheader("**Pie-Chart for State**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based on State 🚉', color_discrete_sequence=px.colors.sequential.PuBu_r)
                st.plotly_chart(fig)

                # fetching Country from the query and getting the unique values and total value count 
                labels = plot_data.Country.unique()
                values = plot_data.Country.value_counts()

                # Pie chart for Country
                st.subheader("**Pie-Chart for Country**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based on Country 🌏', color_discrete_sequence=px.colors.sequential.Purpor_r)
                st.plotly_chart(fig)

                ### NEW: Danger Zone — empty the database directly from within the app.
                ### Reuses the same `connection`/`cursor` already established above,
                ### so no external tool or separate credentials are needed.
                st.markdown("---")
                with st.expander("🛑 Danger Zone — Empty Database", expanded=False):
                    st.warning("This will permanently delete ALL applicant and feedback records. This action cannot be undone.")
                    confirm_text = st.text_input("Type EMPTY to confirm:", key="empty_db_confirm")
                    if st.button("Empty Database Now", key="empty_db_btn"):
                        if confirm_text.strip() == "EMPTY":
                            try:
                                cursor.execute("TRUNCATE TABLE user_data;")
                                cursor.execute("TRUNCATE TABLE user_feedback;")
                                connection.commit()
                                st.success("✅ Database emptied successfully. Both tables are now clean.")
                            except Exception as e:
                                st.error(f"⚠️ Failed to empty database: {e}")
                        else:
                            st.error("Confirmation text did not match. Type EMPTY exactly (case-sensitive) to proceed.")

                st.markdown("---")
                if st.button("Logout", key="admin_logout_btn"):
                    st.session_state['admin_logged_in'] = False
                    st.rerun()

# Calling the main (run()) function to make the whole process run
run()
