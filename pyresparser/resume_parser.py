import os
import multiprocessing as mp
import io
import spacy
import pprint
from spacy.matcher import Matcher
from . import utils


class ResumeParser(object):

    def __init__(  # ✅ صححت من init إلى init
        self,
        resume,
        skills_file=None,
        custom_regex=None
    ):
        # تحميل النموذج بشكل صحيح
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load("en_core_web_sm")
        
        # استخدام نفس النموذج لـ custom_nlp
        self.custom_nlp = self.nlp
        
        self._skills_file = skills_file
        self._custom_regex = custom_regex
        self._matcher = Matcher(self.nlp.vocab)
        self._details = {
            'name': None,
            'email': None,
            'mobile_number': None,
            'skills': None,
            'degree': None,
            'no_of_pages': None,
        }
        
        self._resume = resume
        if not isinstance(self._resume, io.BytesIO):  # ✅ صححت ByteIO إلى BytesIO
            ext = os.path.splitext(self._resume)[1].split('.')[1]
        else:
            ext = self._resume.name.split('.')[1]
        
        self._text_raw = utils.extract_text(self._resume, '.', ext)
        self._text = ''.join(self._text_raw.split())
        self._nlp = self.nlp(self._text)
        self._custom_nlp = self.custom_nlp(self._text_raw)
        self._noun_chunks = list(self._nlp.noun_chunks)
        self._get_basic_details()

    def get_extracted_data(self):
        return self._details  # ✅ صححت من self.details إلى self._details

    def _get_basic_details(self):  # ✅ صححت من __get_basic_details إلى _get_basic_details
        try:
            cust_ent = utils.extract_entities_wih_custom_model(
                self._custom_nlp  # ✅ صححت من __custom_nlp إلى _custom_nlp
            )
        except:
            cust_ent = {}
        
        try:
            name = utils.extract_name(self._nlp, matcher=self._matcher)  # ✅ صححت
        except:
            name = None
            
        email = utils.extract_email(self._text)  # ✅ صححت من __text إلى _text
        mobile = utils.extract_mobile_number(self._text, self._custom_regex)  # ✅ صححت
        
        try:
            skills = utils.extract_skills(
                self._nlp,  # ✅ صححت
                self._noun_chunks,
                self._skills_file
            )
        except:
            skills = None

        try:
            entities = utils.extract_entity_sections_grad(self._text_raw)  # ✅ صححت
        except:
            entities = {}

        # extract name
        try:
            self._details['name'] = cust_ent['Name'][0]
        except (IndexError, KeyError):
            self._details['name'] = name

        # extract email
        self._details['email'] = email

        # extract mobile number
        self._details['mobile_number'] = mobile

        # extract skills
        self._details['skills'] = skills

        # no of pages
        try:
            self._details['no_of_pages'] = utils.get_number_of_pages(self._resume)  # ✅ صححت
        except:
            self._details['no_of_pages'] = 0

        # extract education Degree
        try:
            self._details['degree'] = cust_ent['Degree']
        except KeyError:
            pass

        return


def resume_result_wrapper(resume):
    parser = ResumeParser(resume)
    return parser.get_extracted_data()


if name == 'main':  # ✅ صححت من 'main إلى 'main'
    pool = mp.Pool(mp.cpu_count())

    resumes = []
    data = []
    for root, directories, filenames in os.walk('resumes'):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)

    results = [
        pool.apply_async(
            resume_result_wrapper,
            args=(x,)
        ) for x in resumes
    ]

    results = [p.get() for p in results]

    pprint.pprint(results)
