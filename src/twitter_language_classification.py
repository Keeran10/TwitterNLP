import os
import sys
import re
import math
from itertools import product
import string

class TwitterPost:
    def __init__(self, id, username, language, tweet):
        self.id = id
        self.username = username
        self.language = language
        self.tweet = tweet


# Serves as an interface to call appropriate models
def executeNaiveBayesClassification(V, n, gamma, training, testing):

    twitter_posts = processData(training)

    executeNgram(V, gamma, n, twitter_posts, testing)

# fetch and stores training/testing data in list
def processData(file_path):
    data = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            component = line.split(None, 3)
            if len(component) != 0:
                data.append(
                    # id, username, language, tweet
                    TwitterPost(component[0], component[1], component[2], component[3])
                )
        return data


def executeNgram(V, gamma, n, training, testing):
    testingData = processData(testing)

    # language frequencies denoted by _f
    eu_p, ca_p, gl_p, es_p, en_p, pt_p, language_p = buildNgramModelByVocabulary(V, training, gamma, n)

    for twitterPost in testingData:
        language, probability = detectTweetNgram(V, twitterPost, eu_p, ca_p, gl_p, es_p, en_p, pt_p, language_p, n)
        writeToTraceFile(twitterPost, language, probability, V, n, gamma)


def buildNgramModelByVocabulary(V, twitter_posts, gamma, n):

    if V == 0:
        pattern = re.compile("[a-z]")
        # get dictionary that has all the possible combinations of letters. example when n=3: {aaa:0,aab:0,aac:0 ... ...,zzx:0,zzy:0,zzz:0}
        eu_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)]))  # Basque
        ca_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)]))  # Catalan
        gl_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)]))  # Galician
        es_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)]))  # Spanish
        en_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)]))  # English
        pt_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)])) # Portuguese
        # print(eu_n_gram_dict)
        # print(len(pt_n_gram_dict))
    elif V == 1:
        pattern = re.compile("[A-Za-z]")
        # get dictionary that has all the possible combinations of letters. example when n=3: {aaa:0,aab:0,aac:0 ... ...,zzx:0,zzy:0,zzz:0}
        eu_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # Basque
        ca_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # Catalan
        gl_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # Galician
        es_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # Spanish
        en_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # English
        pt_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # Portuguese
        # print(eu_n_gram_dict)
        # print(len(pt_n_gram_dict))
    elif V == 2:
        pattern = None
        # dont know what the dictionary contains, so adding key value pair later on the go
        eu_n_gram_dict = {}  # Basque
        ca_n_gram_dict = {}  # Catalan
        gl_n_gram_dict = {}  # Galician
        es_n_gram_dict = {}  # Spanish
        en_n_gram_dict = {}  # English
        pt_n_gram_dict = {}  # Portuguese

    language_probability = {'eu': 0 ,'ca': 0 ,'gl': 0 ,'es': 0 ,'en': 0 ,'pt': 0 }
    total_tweets = len(twitter_posts) 

    for twitter_post in twitter_posts:
        if twitter_post.language == "eu":
            language_probability['eu'] = language_probability['eu'] + 1 # for getting the total number of tweets in eu
            buildNgramLanguageModel(eu_n_gram_dict, twitter_post, V, pattern, n) # {"abs": 1, "axs": 1, ...}
        if twitter_post.language == "ca":
            language_probability['ca'] = language_probability['ca'] + 1
            buildNgramLanguageModel(ca_n_gram_dict, twitter_post, V, pattern, n)
        if twitter_post.language == "gl":
            language_probability['gl'] = language_probability['gl'] + 1
            buildNgramLanguageModel(gl_n_gram_dict, twitter_post, V, pattern, n)
        if twitter_post.language == "es":
            language_probability['es'] = language_probability['es'] + 1
            buildNgramLanguageModel(es_n_gram_dict, twitter_post, V, pattern, n)
        if twitter_post.language == "en":
            language_probability['en'] = language_probability['en'] + 1
            buildNgramLanguageModel(en_n_gram_dict, twitter_post, V, pattern, n)
        if twitter_post.language == "pt":
            language_probability['pt'] = language_probability['pt'] + 1
            buildNgramLanguageModel(pt_n_gram_dict, twitter_post, V, pattern, n) 

    for key, value in language_probability.items():
        language_probability[key] = math.log(value / total_tweets) # list: [log(probability of a language showing up)]
    eu_n_gram_dict["NOT-APPEAR"] = 0  # Basque       {"abs": 1, "axs": 1, ..., "NOT-APPEAR": 0}
    ca_n_gram_dict["NOT-APPEAR"] = 0   # Catalan
    gl_n_gram_dict["NOT-APPEAR"] = 0   # Galician
    es_n_gram_dict["NOT-APPEAR"] = 0   # Spanish
    en_n_gram_dict["NOT-APPEAR"] = 0   # English
    pt_n_gram_dict["NOT-APPEAR"] = 0   # Portuguese

    eu_trigram_probability = NgramConditionalProbability(eu_n_gram_dict, gamma) # smoothing if needed, {"abs": (1+0.1)/N+V*0.1, "axs": (1+0.1)/N+V*0.1, ..., "NOT-APPEAR": (0+0.1)/N+V*0.1}
    ca_trigram_probability = NgramConditionalProbability(ca_n_gram_dict, gamma)
    gl_trigram_probability = NgramConditionalProbability(gl_n_gram_dict, gamma)
    es_trigram_probability = NgramConditionalProbability(es_n_gram_dict, gamma)
    en_trigram_probability = NgramConditionalProbability(en_n_gram_dict, gamma)
    pt_trigram_probability = NgramConditionalProbability(pt_n_gram_dict, gamma)

    return (eu_trigram_probability, ca_trigram_probability, gl_trigram_probability, es_trigram_probability, en_trigram_probability, pt_trigram_probability, language_probability)

def buildNgramLanguageModel(ngram_dict, twitter_post, V, pattern, n):
    if V == 0: 
        tweet = twitter_post.tweet.lower()
        for i in range(len(tweet)-n):
            ngram = tweet[i:i+n] # get an ngram, "a", "aa", or "aaa"
            if pattern.match(ngram) is not None:
                if ngram in ngram_dict.keys():
                    ngram_dict[ngram] += 1
                else:
                    ngram_dict[ngram] = 1

    if V == 1:
        tweet = twitter_post.tweet
        for i in range(len(tweet)-n):
            ngram = tweet[i:i+n]
            if pattern.match(ngram) is not None:
                if ngram in ngram_dict.keys():
                    ngram_dict[ngram] += 1
                else:
                    ngram_dict[ngram] = 1

    if V == 2:
        tweet = twitter_post.tweet
        for i in range(len(tweet)-n):
            ngram = tweet[i:i+n]
            if ngram.isalpha() is True:
                if ngram in ngram_dict.keys():
                    ngram_dict[ngram] += 1
                else:
                    ngram_dict[ngram] = 1

def NgramConditionalProbability(ngram_dict, gamma):    
    if gamma != 0 and gamma <= 1:
        total_ngram = len(ngram_dict)

        total_instance = 0
        for key, value in ngram_dict.items():
            total_instance = total_instance + value
            
        for key, value in ngram_dict.items():
            ngram_dict[key] = (value + gamma)/(total_instance + gamma*total_ngram)

    return ngram_dict

def detectTweetNgram(V, twitterPost, eu_p, ca_p, gl_p, es_p, en_p, pt_p, language_p, n):
    probability_dict = {}
    probability_dict['eu'] = language_p['eu'] + ngramCalculateProbability(eu_p, twitterPost, V, n)
    probability_dict['ca'] = language_p['ca'] + ngramCalculateProbability(ca_p, twitterPost, V, n)
    probability_dict['gl'] = language_p['gl'] + ngramCalculateProbability(gl_p, twitterPost, V, n)
    probability_dict['es'] = language_p['es'] + ngramCalculateProbability(es_p, twitterPost, V, n)
    probability_dict['en'] = language_p['en'] + ngramCalculateProbability(en_p, twitterPost, V, n)
    probability_dict['pt'] = language_p['pt'] + ngramCalculateProbability(pt_p, twitterPost, V, n)

    language = max(probability_dict, key=probability_dict.get)

    probability = probability_dict[language]
    return language, probability
    


def ngramCalculateProbability(target_language_p, twitterPost, V, n):
    if V == 0:
        pattern = re.compile("[a-z]")
        tweet = twitterPost.tweet.lower()
        p = 0
        for i in range(len(tweet)-n):
            ngram = tweet[i:i+n]
            if pattern.match(ngram) is not None:
                if ngram in target_language_p.keys():
                    if target_language_p[ngram] == 0:
                        p += -math.inf
                    else:
                        p += math.log(target_language_p[ngram])
                else:
                    p += math.log(target_language_p["NOT-APPEAR"])
        
    if V == 1:
        pattern = re.compile("[A-Za-z]")
        tweet = twitterPost.tweet
        p = 0
        for i in range(len(tweet)-n):
            ngram = tweet[i:i+n]
            if pattern.match(ngram) is not None:
                if ngram in target_language_p.keys():
                    if target_language_p[ngram] == 0:
                        p += -math.inf
                    else:
                        p += math.log(target_language_p[ngram])
                else:
                    p += math.log(target_language_p["NOT-APPEAR"])

    if V == 2:
        tweet = twitterPost.tweet
        p = 0
        for i in range(len(tweet)-n):
            ngram = tweet[i:i+n]
            if ngram.isalpha() is True:
                if ngram in target_language_p.keys():
                    if target_language_p[ngram] == 0:
                        p += -math.inf
                    else:
                        p += math.log(target_language_p[ngram])
                else:
                    p += math.log(target_language_p["NOT-APPEAR"])
    return p
    


# writes final answers in trace file
def writeToTraceFile(twitterPost, language, probability, V, n, gamma):

    if language == twitterPost.language:
        label = "correct"
    else:
        label = "wrong"

    with open(
        "trace_" + str(V) + "_" + str(n) + "_" + str(gamma) + ".txt",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            twitterPost.id
            + "  "
            + language
            + "  "
            + "{:.2e}".format(probability)
            + "  "
            + twitterPost.language
            + "  "
            + label
            + "\n"
        )

# find Basque's diacritics from the training tweets
def BasqueSpecialCharacters():
    data = []
    diacritics = []
    count = 0
    with open(os.path.join(sys.path[0], "training-tweets.txt"), "r", encoding="utf8") as f:
        for line in f:
            component = line.split(None, 3)
            if len(component) != 0 and component[2] == "eu":
                count += 1
                data.append(
                    # id, username, language, tweet
                    TwitterPost(component[0], component[1], component[2], component[3])
                )

    for d in data:
        for c in d.tweet:
            if c not in string.ascii_letters and c.isalpha() and c.lower() not in diacritics:
                diacritics.append(c.lower())
        # print(d.id, d.username, d.language, d.tweet)

    # print("Count:", count)
    # print(diacritics)
    return diacritics

# find Catalan's diacritics from the training tweets
def CatalanSpecialCharacters():
    data = []
    diacritics = []
    count = 0
    with open(os.path.join(sys.path[0], "training-tweets.txt"), "r", encoding="utf8") as f:
        for line in f:
            component = line.split(None, 3)
            if len(component) != 0 and component[2] == "ca":
                count += 1
                data.append(
                    # id, username, language, tweet
                    TwitterPost(component[0], component[1], component[2], component[3])
                )

    for d in data:
        for c in d.tweet:
            if c not in string.ascii_letters and c.isalpha() and c.lower() not in diacritics:
                diacritics.append(c.lower())
        # print(d.id, d.username, d.language, d.tweet)

    # print("Count:", count)
    # print(diacritics)
    return diacritics


# find Galician's diacritics from the training tweets
def GalicianSpecialCharacters():
    data = []
    diacritics = []
    count = 0
    with open(os.path.join(sys.path[0], "training-tweets.txt"), "r", encoding="utf8") as f:
        for line in f:
            component = line.split(None, 3)
            if len(component) != 0 and component[2] == "gl":
                count += 1
                data.append(
                    # id, username, language, tweet
                    TwitterPost(component[0], component[1], component[2], component[3])
                )

    for d in data:
        for c in d.tweet:
            if c not in string.ascii_letters and c.isalpha() and c.lower() not in diacritics:
                diacritics.append(c.lower())
        # print(d.id, d.username, d.language, d.tweet)

    # print("Count:", count)
    # print(diacritics)
    return diacritics

# return Spanish's diacritics according to the reference
def SpanishSpecialCharacters():
    list = ["á", "é", "í", "ó", "ú", "ñ", "ü"]  # reference: paper "Automatic Language Identification for Romance Languages using Stop Words and Diacritics"
    return list

# return portuguese's diacritics according to the reference
def PortugueseSpecialCharacters():
    list = ["â", "ê", "ô", "á", "à", "é", "í", "ó", "ú", "ã", "õ", "ç"] #reference: https://www.fluentu.com/blog/portuguese/portuguese-accent-marks/
    return list


def stopWords(lang):
    if lang == "eu":
        list = ['al','anitz','arabera','asko','baina',"bat","batean","batek","bati","batzuei","batzuek",'batzuetan',
                'batzuk','bera','beraiek','berau','berauek','bere','berori','beroriek','beste','bezala','da','dago',
                'dira','ditu','du','dute','edo','egin','ere','eta','eurak','ez','gainera','gu','gutxi','guzti','haiei',
                'haietan','hainbeste','hala','han','handik','hango','hara','hari','hark','hartan','hau','hauei','hauek',
                'hauetan','hemen','hemendik','hemengo','hi','hona','honek','honela','honetan','honi','hor','hori','horiei',
                'horiek','horietan','horko','horra	horrek','horrela','horretan','horri','hortik','hura','izan','ni','noiz',
                'nola','non','nondik','nongo','nora','ze','zein','zen','zenbait','zenbat','zer','zergatik','ziren','zituen',
                'zu','zuek','zuen','zuten']
        list.sort()
        return list

    elif lang == "ca":
        list = ['de','es','i','a','o','un','una','unes','uns','un','tot','també','altre','algun','alguna','alguns','algunes',
                'ser','és','soc','ets','som','estic','està','estem','esteu','estan','com','en','per','perquè','per que','estat',
                'estava','ans','abans','éssent','ambdós','però','per','poder','potser','puc','podem','podeu','poden','vaig','va',
                'van','fer','faig','fa','fem','feu','fan','cada','fi','inclòs','primer','des de','conseguir','consegueixo','consigueix',
                'consigueixes','conseguim','consigueixen','anar','haver','tenir','tinc','te','tenim','teniu','tene','el','la','les',
                'els','seu','aquí','meu','teu','ells','elles','ens','nosaltres','vosaltres','si','dins','sols','solament','saber','saps',
                'sap','sabem','sabeu','saben','últim','llarg','bastant','fas','molts','aquells','aquelles','seus','llavors','sota','dalt',
                'ús','molt','era','eres','erem','eren','mode','bé','quant','quan','on','mentre','qui','amb','entre','sense','jo','aquell']
        list.sort()
        return list
    elif lang == "gl":
        list = ['a','aínda','alí','aquel','aquela','aquelas','aqueles','aquilo','aquí','ao','aos','as','así','á','ben','cando','che','co',
                  'coa','comigo','con','connosco','contigo','convosco','coas','cos','cun','cuns','cunha','cunhas','da','dalgunha','dalgunhas',
                  'dalgún','dalgúns','das','de','del','dela','delas','deles','desde','deste','do','dos','dun','duns','dunha','dunhas','e','el',
                  'ela','elas','eles','era','eran','esa','esas','ese','eses','esta','estar','estaba','está','este','estes','estiven','estou','eu',
                  'é','facer','foi','foron','fun','había','hai','iso','isto','la','las','lle','lles','lo','los','mais','me','meu','meus','min',
                  'miña','miñas','moi','na','nas','neste','nin','no','non','nos','nosa','nosas','noso','nosos','nós','nun','nunha','nuns','nunhas',
                  'o','os','ou','ó','ós','para','pero','pode','pois','pola','polas','polo','polos','por','que','se','senón','ser','seu','seus','sexa',
                  'sido','sobre','súa','súas','tamén','tan','te','ten','teñen','teño','ter','teu','teus','ti','tido','tiña','tiven','túa','túas','un','unha',
                  'unhas','uns','vos','vosa','vosas','voso','vosos','vós']
        list.sort()
        return list
    elif lang == "es":
        list = ['un','una','unas','unos','uno','sobre','todo','también','tras','otro','algún','alguno','alguna','algunos','algunas','ser','es',
                'soy','eres','somos','sois','estoy','esta','estamos','estais','estan','como','en','para','atras','porque','por qué','estado','estaba',
                'ante','antes','siendo','ambos','pero','por','poder','puede','puedo','podemos','podeis','pueden','fui','fue','fuimos','fueron','hacer',
                'hago','hace','hacemos','haceis','hacen','cada','fin','incluso','primero','desde','conseguir','consigo','consigue','consigues','conseguimos',
                'consiguen','ir','voy','va','vamos','vais','van','vaya','gueno','ha','tener','tiene','tenemos','teneis','tienen','el','la','lo','las',
                'los','su','aqui','mio','tuyo','ellos','ellas','nos','nosotros','vosotros','vosotras','si','dentro','solo','solamente','saber','sabes','sabe',
                'sabemos','sabeis','saben','ultimo','largo','bastante','haces','muchos','aquellos','aquellas','sus','entonces','tiempo','verdad','verdadero',
                'verdadera','cierto','ciertos','cierta','ciertas','intentar','intento','intenta','intentas','intentamos','intentais','intentan','dos','bajo',
                'arriba','encima','usar','uso','usas','usa','usamos','usais','emplear','empleo','empleas','emplean','ampleamos','empleais','valor','muy',
                'era','eras','eramos','eran','modo','bien','cual','cuando','donde','mientras','quien','entre','sin','trabajo','trabajar','trabajas','trabaja',
                'trabajamos','trabajais','trabajan','podria','podrias','podriamos','podrian','podriais','yo','aquel']
        list.sort()
        return list

    elif lang == "en":
        list = ["a","about","above","after","again","against","all","am","an","and","are","aren't","as","at","be","because","been","before","being","below",
                "between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few",
                "for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself",
                "him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most",
                "mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same",
                "shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves",
                "then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was",
                "wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why",
                "why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]
        list.sort()
        return list
    elif lang == "pt":
        list = ["último","é","acerca","agora","algmas","alguns","ali","ambos","antes","apontar","aquela","aquelas","aquele","aqueles","aqui","atrás","bem","bom","cada",
                "caminho","cima","com","como","comprido","conhecido","corrente","das","debaixo","dentro","desde","desligado","deve","devem","deverá","direita","diz","dizer",
                "dois","dos","e","ela","ele","eles","em","enquanto","então","está","estão","estado","estar","estará","este","estes","esteve","estive","estivemos","estiveram",
                "eu","fará","faz","fazer","fazia","fez","fim","foi","fora","horas","iniciar","inicio","ir","irá","ista","iste","isto","ligado","maioria","maiorias","mais",
                "mas","mesmo","meu","muito","muitos","nós","não","nome","nosso","novo","o","onde","os","ou","outro","para","parte","pegar","pelo","pessoas","pode","poderá",
                "podia","por","porque","povo","promeiro","quê","qual","qualquer","quando","quem","quieto","são","saber","sem","ser","seu","somente","têm","tal","também","tem",
                "tempo","tenho","tentar","tentaram","tente","tentei","teu","teve","tipo","tive","todos","trabalhar","trabalho","tu","um","uma","umas","uns","usa","usar",
                "valor","veja","ver","verdade","verdadeiro","você"]
        list.sort()
        return list


def processBYOMData(file_path):
    data = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            component = line.split(None, 3)
            if len(component) != 0:
                data.append(
                    # id, username, language, tweet
                    TwitterPost(component[0], component[1], component[2], re.split('[, .!]', component[3].strip()))
                )

    # for d in data:
    #     print(d.id, d.username, d.language, d.tweet)

    return data


def BYOM(testing):
    stopwords_data = processBYOMData(testing)

    for tweeterPost in stopwords_data:
        if tweeterPost.language == "ca":
            eu_stopwords_f = 0
            ca_stopwords_f = 0
            gl_stopwords_f = 0
            es_stopwords_f = 0
            en_stopwords_f = 0
            pt_stopwords_f = 0
            for token in tweeterPost.tweet:
                if token.lower() in stopWords("eu"):
                    # print ("eu", token)
                    eu_stopwords_f += 1
                if token.lower() in stopWords("ca"):
                    # print ("ca", token)
                    ca_stopwords_f += 1
                if token.lower() in stopWords("gl"):
                    # print ("gl", token)
                    gl_stopwords_f += 1
                if token.lower() in stopWords("es"):
                    # print ("es", token)
                    es_stopwords_f += 1
                if token.lower() in stopWords("en"):
                    # print ("en", token)
                    en_stopwords_f += 1
                if token.lower() in stopWords("pt"):
                    # print ("pt", token)
                    pt_stopwords_f += 1
            print("stopwords", eu_stopwords_f, ca_stopwords_f, gl_stopwords_f, es_stopwords_f, en_stopwords_f, pt_stopwords_f)




def main():
    # for V in range(3):
    #     for n in range(1,4):
    #         gamma = 0.1
    #         print("V=%d, n=%d, gamma=%f" % (V,n,gamma))
    #         # training = os.path.join(sys.path[0], "training-tweets.txt")
    #         training = os.path.join(sys.path[0], "training-tweets.txt")
    #         testing = os.path.join(sys.path[0], "test-tweets-given.txt")
    #         executeNaiveBayesClassification(V, n, gamma, training, testing)
    BYOM(os.path.join(sys.path[0], "test-tweets-given.txt"))
    # print(stopWords("pt"))

if __name__ == "__main__":
    main()




