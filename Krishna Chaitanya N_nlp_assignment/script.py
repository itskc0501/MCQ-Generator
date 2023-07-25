# Import all the dependent libraries
# import os
import random
import re  # for regular expression

import PyPDF2  # for reading pdf files
from summarizer import Summarizer  # Bert Model Summarizer

import pke  # for keyphrase extraction
# !python -m spacy download en - inbuilt
# import nltk

from flashtext import KeywordProcessor  # for keyphrase extraction
import spacy

import requests  # for web scrapping


def get_mca_questions(context: str):
    # The str is read under __main__.py
    # By default, two questions per dataset will be generated.
    mca_questions = ["mca1", "mca2"]

    model = GenerateMCQ(str, len(mca_questions))
    questions = model.generate()
    mca_questions = questions

    return mca_questions


def read_pdf(path):
    with open(path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)  # reads binary output
        txt = ''
        for page in pdf.pages:
            txt += page.extract_text().replace('\n', ' ')
    return txt


class GenerateMCQ:
    def __init__(self, text, num_quest=2):
        self.text = text
        self.num_quest = num_quest
        random.seed(79)

    def summarize(self):
        model = Summarizer()
        summary = model(self.text, max_length=500, min_length=50, ratio=0.4)
        summary = ''.join(summary)
        return summary

    def keywords(self, sum_text):
        key = []
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection(pos={'PROPN'})
        extractor.candidate_weighting(
            alpha=1.1, threshold=0.75, method='average')
        phrases = extractor.get_n_best(n=20)
        for k in phrases:
            key.append(k[0])

        keys = []
        for k in key:
            if k.lower() in sum_text.lower():
                keys.append(k)
        return keys

    def key_sent(self, sum_text, keys):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sum_text)
        sentences = [sent.text.strip() for sent in doc.sents]

        kw_processor = KeywordProcessor()
        keysent = {}
        for key in keys:
            keysent[key] = []
            kw_processor.add_keyword(key)
        for sent in sentences:
            for keyword in kw_processor.extract_keywords(sent):
                keysent[keyword].append(sent)
        for key in keysent.keys():
            val = keysent[key]
            val = sorted(val, key=len, reverse=True)
            keysent[key] = val
        return keysent

    def distractors(self, word):
        # Creation of distractors using ConceptNet
        dist = []
        url = "http://api.conceptnet.io/"
        url = url + "query?start=/c/en/" + word + "&rel=/r/RelatedTo&limit=10"
        response = requests.get(url, timeout=30).json()
        for edge in response['edges']:
            end_node = edge['end']['label']
            if end_node.lower() != word.lower():
                dist.append(end_node)
        return dist

    def form_quest(self, key_sent):
        quest_ = []
        for word1 in key_sent:
            for word2 in key_sent:
                if word1 != word2:
                    w_1 = key_sent[word1]
                    w_2 = key_sent[word2]
                    for s_a in w_1:
                        for s_b in w_2:
                            if s_a == s_b:
                                quest_.append((word1, word2, s_a))
        return quest_

    def list_quest(self, quest_):
        questions = []
        for _ in range(self.num_quest):
            question = random.choice(quest_)
            word1 = question[0]
            word2 = question[1]
            sentence = question[2]

            pattern = re.compile(f'{word1}|{word2}', re.IGNORECASE)
            output = pattern.sub(' __________ ', sentence)

            str_output = 'Q. ' + output + '\n'
            choices = [word1.capitalize()] + [word2.capitalize()] + \
                [self.distractors(word1)[0]] + [self.distractors(word2)[0]]
            random.shuffle(choices)
            output_choices = ['Option A:',
                              'Option B:', 'Option C:', 'Option D:']
            for idx, choice in enumerate(choices):
                str_output += output_choices[idx] + ' ' + choice + '\n'
            min_val = min(choices.index(word1.capitalize()),
                          choices.index(word2.capitalize()))
            max_val = max(choices.index(word1.capitalize()),
                          choices.index(word2.capitalize()))
            str_output += f'The Correct Options are: {output_choices[min_val]} and {output_choices[max_val]}.\n'
            questions.append(str_output)
        return questions

    def generate(self):
        sum_text = self.summarize()
        keys = self.keywords(sum_text)
        keysent = self.key_sent(sum_text, keys)
        quest_ = self.form_quest(keysent)

        questions = self.list_quest(quest_)
        return questions


if __name__ == '__main__':
    # Read all the text from the pdf files given.
    chap_2 = read_pdf('Dataset\chapter-2.pdf')
    chap_3 = read_pdf('Dataset\chapter-3.pdf')
    chap_4 = read_pdf('Dataset\chapter-4.pdf')

    # Preprocess the text
    UNDESIRED1 = 'chap 1-4.indd   23 4/22/2022   2:49:43 PMRationalised 2023-24 24 OUR PASTS – III Let’s recall 1. Match the following: Diwani Tipu Sultan “Tiger of Mysore” right to collect land revenue  faujdari adalat  Sepoy   Rani Channamma criminal court sipahi  led an anti-British   movement in Kitoor  2. Fill in the blanks: (a) The British conquest of Bengal began with the  Battle of ___________. (b) Haidar Ali and Tipu Sultan were the rulers of   ___________. (c) Dalhousie implemented the Doctrine of   ___________.  (d) Maratha kingdoms were located mainly in the   ___________ part of India. 3. State whether true or false: (a) The Mughal empire became stronger in the eighteenth century. (b) The English East India Company was the only  European company that traded with India. (c) Maharaja Ranjit Singh was the ruler of Punjab. (d) The British did not introduce administrative   changes in the territories they conquered. Let’s imagine You are living in  England in the late eighteenth or early nineteenth century. How would you have reacted to the stories of British conquests? Remember that you would have read about the immense fortunes that many of the officials were making. Let’s discuss 4. What attracted European trading companies to  India? 5. What were the areas of conflict between the  Bengal nawabs and the East India Company? chap 1-4.indd   24 4/22/2022   2:49:46 PMRationalised 2023-24 FROM TRADE TO TERRITORY         25 6. How did the assumption of Diwani benefit the  East India Company? 7. Explain the system of “subsidiary alliance”. 8. In what way was the administration of the  Company different from that of Indian rulers? 9. Describe the changes that occurred in the   composition of the Company’s army. Let’s do 10. After the British conquest of Bengal, Calcutta  grew from small village to a big city. Find out about the culture, architecture and the life of Europeans and Indians of the city during the colonial period. 11. Collect pictures, stories, poems and information  about any of the following – the Rani of Jhansi, Mahadji Sindhia, Haidar Ali, Maharaja Ranjit Singh, Lord Dalhousie or any other contemporary ruler of your region. chap 1-4.indd   25 4/22/2022   2:49:46 PMRationalised 2023-24'
    chap_2 = chap_2.replace(UNDESIRED1, '')
    chap_2 = chap_2.replace('FROM TRADE TO TERRITORY', '')
    UNDESIRED2 = 'From Trade to Territory                    The Company Establishes Power2'
    chap_2 = chap_2.replace(UNDESIRED2, '')
    chap_2 = chap_2.replace('OUR PASTS - III', '')  # OUR PASTS – III
    chap_2 = chap_2.replace('PMRationalised 2023-24',
                            '')  # PMRationalised 2023-24
    chap_2 = chap_2.replace('chap 1-4.indd', '')

    UNDESIRED1 = 'chap 1-4.indd   37 4/22/2022   2:50:01 PMRationalised 2023-24 38 OUR PASTS – III Let’s imagine Imagine a conversation  between a planter and a peasant who is being forced to grow indigo. What reasons would the planter give to persuade the peasant? What problems would the peasant point out? Enact their conversation.5. Give two problems which arose with the new Munro  system of fixing revenue. 6. Why were ryots reluctant to grow indigo? 7. What were the circumstances which led to the  eventual collapse of indigo production in Bengal?  Let’s do 8. Find out more about the Champaran movement and   Mahatma Gandhi’s role in it.  9. Look into the history of either tea or coffee plantations in India. See how the life of workers in  these plantations was similar to or different from that of workers in indigo plantations. chap 1-4.indd   38 4/22/2022   2:50:03 PMRationalised 2023-24'
    chap_3 = chap_3.replace(UNDESIRED1, '')
    UNDESIRED2 = 'chap 1-4.indd   36 4/22/2022   2:50:00 PMRationalised 2023-24  RULING THE COUNTRYSIDE          37 Let’s recall 1. Match the following: ryot  village mahal peasantnij  cultivation on ryot’s lands ryoti  cultivation on planter’s own land 2. Fill in the blanks: (a) Growers of woad in Europe saw __________  as a crop which would provide competition to their earnings.  (b) The demand for indigo increased in late eighteenth-century Britain because of __________. (c) The international demand for indigo was affected by the discovery of __________. (d) The Champaran movement was against __________. Let’s discuss 3. Describe the main features of the Permanent Settlement.  4. How was the mahalwari system different from the  Permanent Settlement?'
    chap_3 = chap_3.replace(UNDESIRED2, '')
    chap_3 = chap_3.replace('RULING THE COUNTRYSIDE', '')
    UNDESIRED3 = 'Ruling the Countryside                   3 chap 1-4.indd   26 4/22/2022   2:49:47 PMRationalised 2023-24'
    chap_3 = chap_3.replace(UNDESIRED3, '')
    chap_3 = chap_3.replace('OUR PASTS – III', '')
    chap_3 = chap_3.replace('PMRationalised 2023-24', '')
    chap_3 = chap_3.replace('chap 1-4.indd', '')

    UNDESIRED1 = 'Let’s recall 1. Fill in the blanks: (a) The British described the tribal people as ____________. (b) The method of sowing seeds in jhum cultivation  is known as ____________. (c) The tribal chiefs got _________ titles in central  India under the British land settlements. (d) Tribals went to work in the __________ of Assam  and the ____________ in Bihar.   chap 1-4.indd   49 4/22/2022   2:50:17 PMRationalised 2023-24 50 OUR PASTS – III 2. State whether true or false: (a) Jhum cultivators plough the land and sow seeds. (b) Cocoons were bought from the Santhals and  sold  by the traders at five times the purchase price. (c) Birsa urged his followers to purify themselves,  give up drinking liquor and stop believing in witchcraft and sorcery. (d) The British wanted to preserve the tribal way of  life. Let’s discuss 3. What problems did  shifting cultivators face under  British rule? 4. How did the powers of tribal chiefs change under   colonial rule? 5. What accounts for the anger of the tribals against   the dikus? 6.  What was Birsa’s vision of a golden age? Why do   you think such a vision appealed to the people of the region? Let’s do 7. Find out from your parents, friends or teachers, the names of some heroes of other tribal revolts in the twentieth century. Write their story in your  own words. 8. Choose any tribal group living in India today. Find  out about their customs and way of life, and how their lives have changed in the last 50 years.Let’s imagine Imagine you are a  jhum cultivator living in a forest village in the nineteenth century. You have just been told that the land you were born on no longer belongs to you. In a meeting with British officials you try to explain the kinds of problems you face. What would you say?  chap 1-4.indd   50 4/22/2022   2:50:18 PMRationalised 2023-24'
    chap_4 = chap_4.replace(UNDESIRED1, '')
    chap_4 = chap_4.replace(
        'TRIBALS, DIKUS AND THE VISION OF A GOLDEN AGE', '')
    UNDESIRED2 = 'Tribals, Dikus and the   Vision of a Golden Age4 '
    chap_4 = chap_4.replace(UNDESIRED2, '')
    chap_4 = chap_4.replace('OUR PASTS – III', '')
    chap_4 = chap_4.replace('PMRationalised 2023-24', '')
    chap_4 = chap_4.replace('chap 1-4.indd', '')

    # Get the required questions:
    chap_2_quest = get_mca_questions(chap_2)
    chap_3_quest = get_mca_questions(chap_3)
    chap_4_quest = get_mca_questions(chap_4)
