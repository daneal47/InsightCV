import os
import multiprocessing as mp
import io
import spacy
import pprint
from spacy.matcher import Matcher
from . import utils


class ResumeParser(object):

   def init(
    self,
    resume,
    skills_file=None,
    custom_regex=None
):
    nlp = spacy.load("en_core_web_sm")
    try:
        custom_nlp = spacy.load("en_core_web_sm")
    except:
        custom_nlp = nlp
    
    self._skills_file = skills_file
    self._custom_regex = custom_regex
    self._matcher = Matcher(nlp.vocab)
    self._details = {
        'name': None,
        'email': None,
        'mobile_number': None,
        'skills': None,
        'degree': None,
        'no_of_pages': None,
    }
    
    self._resume = resume
    if not isinstance(self._resume, io.ByteIO):
        ext = os.path.splitext(self._resume)[1].split('.')[1]  # ✅
    else:
        ext = self._resume.name.split('.')[1]
    
    self._text_raw = utils.extract_text(self._resume, '.', ext)  # ✅
    self._text = ''.join(self._text_raw.split())
    self._nlp = nlp(self._text)
    self._custom_nlp = custom_nlp(self._text_raw)
    self._noun_chunks = list(self._nlp.noun_chunks)
    self._get_basic_details()

    def get_extracted_data(self):
        return self.__details

    def __get_basic_details(self):
        cust_ent = utils.extract_entities_wih_custom_model(
                            self.__custom_nlp
                        )
        name = utils.extract_name(self.__nlp, matcher=self.__matcher)
        email = utils.extract_email(self.__text)
        mobile = utils.extract_mobile_number(self.__text, self.__custom_regex)
        skills = utils.extract_skills(
                    self.__nlp,
                    self.__noun_chunks,
                    self.__skills_file
                )

        entities = utils.extract_entity_sections_grad(self.__text_raw)

        # extract name
        try:
            self.__details['name'] = cust_ent['Name'][0]
        except (IndexError, KeyError):
            self.__details['name'] = name

        # extract email
        self.__details['email'] = email

        # extract mobile number
        self.__details['mobile_number'] = mobile

        # extract skills
        self.__details['skills'] = skills

        # no of pages
        self.__details['no_of_pages'] = utils.get_number_of_pages(self.__resume)

        # extract education Degree
        try:
            self.__details['degree'] = cust_ent['Degree']
        except KeyError:
            pass

        return


def resume_result_wrapper(resume):
    parser = ResumeParser(resume)
    return parser.get_extracted_data()


if __name__ == '__main__':
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
