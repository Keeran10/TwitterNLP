import os
import sys
import re
import copy
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
    if V == 3:
        # main idea: only keep stopwords in tweets.
        stop_words_toggle = False
        print_toggle = False
        # language is unkown in test set, if one word is in any of 6 stopwords set, it is kept in the tweet. In training set, only the words that are in the stopwords set of their own language are kept.
        is_test = False

        for t in twitter_posts:        
            t.tweet = tweet_preprocess(t, stop_words_toggle, print_toggle, is_test)

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

def tweet_preprocess(t, stop_words_toggle, print_toggle, is_test):
    #stop words for the candidate language.
    eu = set(["al","anitz","arabera","asko","baina","bat","batean","batek","bati","batzuei","batzuek","batzuetan","batzuk","bera","beraiek","berau","berauek","bere","berori","beroriek","beste","bezala","da","dago","dira","ditu","du","dute","edo","egin","ere","eta","eurak","ez","gainera","gu","gutxi","guzti","haiei","haiek","haietan","hainbeste","hala","han","handik","hango","hara","hari","hark","hartan","hau","hauei","hauek","hauetan","hemen","hemendik","hemengo","hi","hona","honek","honela","honetan","honi","hor","hori","horiei","horiek","horietan","horko","horra","horrek","horrela","horretan","horri","hortik","hura","izan","ni","noiz","nola","non","nondik","nongo","nor","nora","ze","zein","zen","zenbait","zenbat","zer","zergatik","ziren","zituen","zu","zuek","zuen","zuten"])  
    ca = set(["a","abans","ací","ah","així","això","al","aleshores","algun","alguna","algunes","alguns","alhora","allà","allí","allò","als","altra","altre","altres","amb","ambdues","ambdós","anar","ans","apa","aquell","aquella","aquelles","aquells","aquest","aquesta","aquestes","aquests","aquí","baix","bastant","bé","cada","cadascuna","cadascunes","cadascuns","cadascú","com","consegueixo","conseguim","conseguir","consigueix","consigueixen","consigueixes","contra","d'un","d'una","d'unes","d'uns","dalt","de","del","dels","des","des de","després","dins","dintre","donat","doncs","durant","e","eh","el","elles","ells","els","em","en","encara","ens","entre","era","erem","eren","eres","es","esta","estan","estat","estava","estaven","estem","esteu","estic","està","estàvem","estàveu","et","etc","ets","fa","faig","fan","fas","fem","fer","feu","fi","fins","fora","gairebé","ha","han","has","haver","havia","he","hem","heu","hi","ho","i","igual","iguals","inclòs","ja","jo","l'hi","la","les","li","li'n","llarg","llavors","m'he","ma","mal","malgrat","mateix","mateixa","mateixes","mateixos","me","mentre","meu","meus","meva","meves","mode","molt","molta","moltes","molts","mon","mons","més","n'he","n'hi","ne","ni","no","nogensmenys","només","nosaltres","nostra","nostre","nostres","o","oh","oi","on","pas","pel","pels","per","per que","perquè","però","poc","poca","pocs","podem","poden","poder","podeu","poques","potser","primer","propi","puc","qual","quals","quan","quant","que","quelcom","qui","quin","quina","quines","quins","què","s'ha","s'han","sa","sabem","saben","saber","sabeu","sap","saps","semblant","semblants","sense","ser","ses","seu","seus","seva","seves","si","sobre","sobretot","soc","solament","sols","som","son","sons","sota","sou","sóc","són","t'ha","t'han","t'he","ta","tal","també","tampoc","tan","tant","tanta","tantes","te","tene","tenim","tenir","teniu","teu","teus","teva","teves","tinc","ton","tons","tot","tota","totes","tots","un","una","unes","uns","us","va","vaig","vam","van","vas","veu","vosaltres","vostra","vostre","vostres","érem","éreu","és","éssent","últim","ús"])
    gl = set(["a","aínda","alí","aquel","aquela","aquelas","aqueles","aquilo","aquí","ao","aos","as","así","á","ben","cando","che","co","coa","comigo","con","connosco","contigo","convosco","coas","cos","cun","cuns","cunha","cunhas","da","dalgunha","dalgunhas","dalgún","dalgúns","das","de","del","dela","delas","deles","desde","deste","do","dos","dun","duns","dunha","dunhas","e","el","ela","elas","eles","en","era","eran","esa","esas","ese","eses","esta","estar","estaba","está","están","este","estes","estiven","estou","eu","é","facer","foi","foron","fun","había","hai","iso","isto","la","las","lle","lles","lo","los","mais","me","meu","meus","min","miña","miñas","moi","na","nas","neste","nin","no","non","nos","nosa","nosas","noso","nosos","nós","nun","nunha","nuns","nunhas","o","os","ou","ó","ós","para","pero","pode","pois","pola","polas","polo","polos","por","que","se","senón","ser","seu","seus","sexa","sido","sobre","súa","súas","tamén","tan","te","ten","teñen","teño","ter","teu","teus","ti","tido","tiña","tiven","túa","túas","un","unha","unhas","uns","vos","vosa","vosas","voso","vosos","vós"])
    es = set(["a","actualmente","acuerdo","adelante","ademas","además","adrede","afirmó","agregó","ahi","ahora","ahí","al","algo","alguna","algunas","alguno","algunos","algún","alli","allí","alrededor","ambos","ampleamos","antano","antaño","ante","anterior","antes","apenas","aproximadamente","aquel","aquella","aquellas","aquello","aquellos","aqui","aquél","aquélla","aquéllas","aquéllos","aquí","arriba","arribaabajo","aseguró","asi","así","atras","aun","aunque","ayer","añadió","aún","b","bajo","bastante","bien","breve","buen","buena","buenas","bueno","buenos","c","cada","casi","cerca","cierta","ciertas","cierto","ciertos","cinco","claro","comentó","como","con","conmigo","conocer","conseguimos","conseguir","considera","consideró","consigo","consigue","consiguen","consigues","contigo","contra","cosas","creo","cual","cuales","cualquier","cuando","cuanta","cuantas","cuanto","cuantos","cuatro","cuenta","cuál","cuáles","cuándo","cuánta","cuántas","cuánto","cuántos","cómo","d","da","dado","dan","dar","de","debajo","debe","deben","debido","decir","dejó","del","delante","demasiado","demás","dentro","deprisa","desde","despacio","despues","después","detras","detrás","dia","dias","dice","dicen","dicho","dieron","diferente","diferentes","dijeron","dijo","dio","donde","dos","durante","día","días","dónde","e","ejemplo","el","ella","ellas","ello","ellos","embargo","empleais","emplean","emplear","empleas","empleo","en","encima","encuentra","enfrente","enseguida","entonces","entre","era","erais","eramos","eran","eras","eres","es","esa","esas","ese","eso","esos","esta","estaba","estabais","estaban","estabas","estad","estada","estadas","estado","estados","estais","estamos","estan","estando","estar","estaremos","estará","estarán","estarás","estaré","estaréis","estaría","estaríais","estaríamos","estarían","estarías","estas","este","estemos","esto","estos","estoy","estuve","estuviera","estuvierais","estuvieran","estuvieras","estuvieron","estuviese","estuvieseis","estuviesen","estuvieses","estuvimos","estuviste","estuvisteis","estuviéramos","estuviésemos","estuvo","está","estábamos","estáis","están","estás","esté","estéis","estén","estés","ex","excepto","existe","existen","explicó","expresó","f","fin","final","fue","fuera","fuerais","fueran","fueras","fueron","fuese","fueseis","fuesen","fueses","fui","fuimos","fuiste","fuisteis","fuéramos","fuésemos","g","general","gran","grandes","gueno","h","ha","haber","habia","habida","habidas","habido","habidos","habiendo","habla","hablan","habremos","habrá","habrán","habrás","habré","habréis","habría","habríais","habríamos","habrían","habrías","habéis","había","habíais","habíamos","habían","habías","hace","haceis","hacemos","hacen","hacer","hacerlo","haces","hacia","haciendo","hago","han","has","hasta","hay","haya","hayamos","hayan","hayas","hayáis","he","hecho","hemos","hicieron","hizo","horas","hoy","hube","hubiera","hubierais","hubieran","hubieras","hubieron","hubiese","hubieseis","hubiesen","hubieses","hubimos","hubiste","hubisteis","hubiéramos","hubiésemos","hubo","i","igual","incluso","indicó","informo","informó","intenta","intentais","intentamos","intentan","intentar","intentas","intento","ir","j","junto","k","l","la","lado","largo","las","le","lejos","les","llegó","lleva","llevar","lo","los","luego","lugar","m","mal","manera","manifestó","mas","mayor","me","mediante","medio","mejor","mencionó","menos","menudo","mi","mia","mias","mientras","mio","mios","mis","misma","mismas","mismo","mismos","modo","momento","mucha","muchas","mucho","muchos","muy","más","mí","mía","mías","mío","míos","n","nada","nadie","ni","ninguna","ningunas","ninguno","ningunos","ningún","no","nos","nosotras","nosotros","nuestra","nuestras","nuestro","nuestros","nueva","nuevas","nuevo","nuevos","nunca","o","ocho","os","otra","otras","otro","otros","p","pais","para","parece","parte","partir","pasada","pasado","paìs","peor","pero","pesar","poca","pocas","poco","pocos","podeis","podemos","poder","podria","podriais","podriamos","podrian","podrias","podrá","podrán","podría","podrían","poner","por","por qué","porque","posible","primer","primera","primero","primeros","principalmente","pronto","propia","propias","propio","propios","proximo","próximo","próximos","pudo","pueda","puede","pueden","puedo","pues","q","qeu","que","quedó","queremos","quien","quienes","quiere","quiza","quizas","quizá","quizás","quién","quiénes","qué","r","raras","realizado","realizar","realizó","repente","respecto","s","sabe","sabeis","sabemos","saben","saber","sabes","sal","salvo","se","sea","seamos","sean","seas","segun","segunda","segundo","según","seis","ser","sera","seremos","será","serán","serás","seré","seréis","sería","seríais","seríamos","serían","serías","seáis","señaló","si","sido","siempre","siendo","siete","sigue","siguiente","sin","sino","sobre","sois","sola","solamente","solas","solo","solos","somos","son","soy","soyos","su","supuesto","sus","suya","suyas","suyo","suyos","sé","sí","sólo","t","tal","tambien","también","tampoco","tan","tanto","tarde","te","temprano","tendremos","tendrá","tendrán","tendrás","tendré","tendréis","tendría","tendríais","tendríamos","tendrían","tendrías","tened","teneis","tenemos","tener","tenga","tengamos","tengan","tengas","tengo","tengáis","tenida","tenidas","tenido","tenidos","teniendo","tenéis","tenía","teníais","teníamos","tenían","tenías","tercera","ti","tiempo","tiene","tienen","tienes","toda","todas","todavia","todavía","todo","todos","total","trabaja","trabajais","trabajamos","trabajan","trabajar","trabajas","trabajo","tras","trata","través","tres","tu","tus","tuve","tuviera","tuvierais","tuvieran","tuvieras","tuvieron","tuviese","tuvieseis","tuviesen","tuvieses","tuvimos","tuviste","tuvisteis","tuviéramos","tuviésemos","tuvo","tuya","tuyas","tuyo","tuyos","tú","u","ultimo","un","una","unas","uno","unos","usa","usais","usamos","usan","usar","usas","uso","usted","ustedes","v","va","vais","valor","vamos","van","varias","varios","vaya","veces","ver","verdad","verdadera","verdadero","vez","vosotras","vosotros","voy","vuestra","vuestras","vuestro","vuestros","w","x","y","ya","yo","z","él","éramos","ésa","ésas","ése","ésos","ésta","éstas","éste","éstos","última","últimas","último","últimos"])
    en = set(["'ll","'tis","'twas","'ve","a","a's","able","ableabout","about","above","abroad","abst","accordance","according","accordingly","across","act","actually","ad","added","adj","adopted","ae","af","affected","affecting","affects","after","afterwards","ag","again","against","ago","ah","ahead","ai","ain't","aint","al","all","allow","allows","almost","alone","along","alongside","already","also","although","always","am","amid","amidst","among","amongst","amoungst","amount","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","ao","apart","apparently","appear","appreciate","appropriate","approximately","aq","ar","are","area","areas","aren","aren't","arent","arise","around","arpa","as","aside","ask","asked","asking","asks","associated","at","au","auth","available","aw","away","awfully","az","b","ba","back","backed","backing","backs","backward","backwards","bb","bd","be","became","because","become","becomes","becoming","been","before","beforehand","began","begin","beginning","beginnings","begins","behind","being","beings","believe","below","beside","besides","best","better","between","beyond","bf","bg","bh","bi","big","bill","billion","biol","bj","bm","bn","bo","both","bottom","br","brief","briefly","bs","bt","but","buy","bv","bw","by","bz","c","c'mon","c's","ca","call","came","can","can't","cannot","cant","caption","case","cases","cause","causes","cc","cd","certain","certainly","cf","cg","ch","changes","ci","ck","cl","clear","clearly","click","cm","cmon","cn","co","co.","com","come","comes","computer","con","concerning","consequently","consider","considering","contain","containing","contains","copy","corresponding","could","could've","couldn","couldn't","couldnt","course","cr","cry","cs","cu","currently","cv","cx","cy","cz","d","dare","daren't","darent","date","de","dear","definitely","describe","described","despite","detail","did","didn","didn't","didnt","differ","different","differently","directly","dj","dk","dm","do","does","doesn","doesn't","doesnt","doing","don","don't","done","dont","doubtful","down","downed","downing","downs","downwards","due","during","dz","e","each","early","ec","ed","edu","ee","effect","eg","eh","eight","eighty","either","eleven","else","elsewhere","empty","end","ended","ending","ends","enough","entirely","er","es","especially","et","et-al","etc","even","evenly","ever","evermore","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","face","faces","fact","facts","fairly","far","farther","felt","few","fewer","ff","fi","fifteen","fifth","fifty","fify","fill","find","finds","fire","first","five","fix","fj","fk","fm","fo","followed","following","follows","for","forever","former","formerly","forth","forty","forward","found","four","fr","free","from","front","full","fully","further","furthered","furthering","furthermore","furthers","fx","g","ga","gave","gb","gd","ge","general","generally","get","gets","getting","gf","gg","gh","gi","give","given","gives","giving","gl","gm","gmt","gn","go","goes","going","gone","good","goods","got","gotten","gov","gp","gq","gr","great","greater","greatest","greetings","group","grouped","grouping","groups","gs","gt","gu","gw","gy","h","had","hadn't","hadnt","half","happens","hardly","has","hasn","hasn't","hasnt","have","haven","haven't","havent","having","he","he'd","he'll","he's","hed","hell","hello","help","hence","her","here","here's","hereafter","hereby","herein","heres","hereupon","hers","herself","herse”","hes","hi","hid","high","higher","highest","him","himself","himse”","his","hither","hk","hm","hn","home","homepage","hopefully","how","how'd","how'll","how's","howbeit","however","hr","ht","htm","html","http","hu","hundred","i","i'd","i'll","i'm","i've","i.e.","id","ie","if","ignored","ii","il","ill","im","immediate","immediately","importance","important","in","inasmuch","inc","inc.","indeed","index","indicate","indicated","indicates","information","inner","inside","insofar","instead","int","interest","interested","interesting","interests","into","invention","inward","io","iq","ir","is","isn","isn't","isnt","it","it'd","it'll","it's","itd","itll","its","itself","itse”","ive","j","je","jm","jo","join","jp","just","k","ke","keep","keeps","kept","keys","kg","kh","ki","kind","km","kn","knew","know","known","knows","kp","kr","kw","ky","kz","l","la","large","largely","last","lately","later","latest","latter","latterly","lb","lc","least","length","less","lest","let","let's","lets","li","like","liked","likely","likewise","line","little","lk","ll","long","longer","longest","look","looking","looks","low","lower","lr","ls","lt","ltd","lu","lv","ly","m","ma","made","mainly","make","makes","making","man","many","may","maybe","mayn't","maynt","mc","md","me","mean","means","meantime","meanwhile","member","members","men","merely","mg","mh","microsoft","might","might've","mightn't","mightnt","mil","mill","million","mine","minus","miss","mk","ml","mm","mn","mo","more","moreover","most","mostly","move","mp","mq","mr","mrs","ms","msie","mt","mu","much","mug","must","must've","mustn't","mustnt","mv","mw","mx","my","myself","myse”","mz","n","na","name","namely","nay","nc","nd","ne","near","nearly","necessarily","necessary","need","needed","needing","needn't","neednt","needs","neither","net","netscape","never","neverf","neverless","nevertheless","new","newer","newest","next","nf","ng","ni","nine","ninety","nl","no","no-one","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","notwithstanding","novel","now","nowhere","np","nr","nu","null","number","numbers","nz","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","older","oldest","om","omitted","on","once","one","one's","ones","only","onto","open","opened","opening","opens","opposite","or","ord","order","ordered","ordering","orders","org","other","others","otherwise","ought","oughtn't","oughtnt","our","ours","ourselves","out","outside","over","overall","owing","own","p","pa","page","pages","part","parted","particular","particularly","parting","parts","past","pe","per","perhaps","pf","pg","ph","pk","pl","place","placed","places","please","plus","pm","pmid","pn","point","pointed","pointing","points","poorly","possible","possibly","potentially","pp","pr","predominantly","present","presented","presenting","presents","presumably","previously","primarily","probably","problem","problems","promptly","proud","provided","provides","pt","put","puts","pw","py","q","qa","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","reasonably","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","reserved","respectively","resulted","resulting","results","right","ring","ro","room","rooms","round","ru","run","rw","s","sa","said","same","saw","say","saying","says","sb","sc","sd","se","sec","second","secondly","seconds","section","see","seeing","seem","seemed","seeming","seems","seen","sees","self","selves","sensible","sent","serious","seriously","seven","seventy","several","sg","sh","shall","shan't","shant","she","she'd","she'll","she's","shed","shell","shes","should","should've","shouldn","shouldn't","shouldnt","show","showed","showing","shown","showns","shows","si","side","sides","significant","significantly","similar","similarly","since","sincere","site","six","sixty","sj","sk","sl","slightly","sm","small","smaller","smallest","sn","so","some","somebody","someday","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","sr","st","state","states","still","stop","strongly","su","sub","substantially","successfully","such","sufficiently","suggest","sup","sure","sv","sy","system","sz","t","t's","take","taken","taking","tc","td","tell","ten","tends","test","text","tf","tg","th","than","thank","thanks","thanx","that","that'll","that's","that've","thatll","thats","thatve","the","their","theirs","them","themselves","then","thence","there","there'd","there'll","there're","there's","there've","thereafter","thereby","thered","therefore","therein","therell","thereof","therere","theres","thereto","thereupon","thereve","these","they","they'd","they'll","they're","they've","theyd","theyll","theyre","theyve","thick","thin","thing","things","think","thinks","third","thirty","this","thorough","thoroughly","those","thou","though","thoughh","thought","thoughts","thousand","three","throug","through","throughout","thru","thus","til","till","tip","tis","tj","tk","tm","tn","to","today","together","too","took","top","toward","towards","tp","tr","tried","tries","trillion","truly","try","trying","ts","tt","turn","turned","turning","turns","tv","tw","twas","twelve","twenty","twice","two","tz","u","ua","ug","uk","um","un","under","underneath","undoing","unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","upwards","us","use","used","useful","usefully","usefulness","uses","using","usually","uucp","uy","uz","v","va","value","various","vc","ve","versus","very","vg","vi","via","viz","vn","vol","vols","vs","vu","w","want","wanted","wanting","wants","was","wasn","wasn't","wasnt","way","ways","we","we'd","we'll","we're","we've","web","webpage","website","wed","welcome","well","wells","went","were","weren","weren't","werent","weve","wf","what","what'd","what'll","what's","what've","whatever","whatll","whats","whatve","when","when'd","when'll","when's","whence","whenever","where","where'd","where'll","where's","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","whichever","while","whilst","whim","whither","who","who'd","who'll","who's","whod","whoever","whole","wholl","whom","whomever","whos","whose","why","why'd","why'll","why's","widely","width","will","willing","wish","with","within","without","won","won't","wonder","wont","words","work","worked","working","works","world","would","would've","wouldn","wouldn't","wouldnt","ws","www","x","y","ye","year","years","yes","yet","you","you'd","you'll","you're","you've","youd","youll","young","younger","youngest","your","youre","yours","yourself","yourselves","youve","yt","yu","z","za","zero","zm","zr"])
    pt = set(["a","acerca","adeus","agora","ainda","alem","algmas","algo","algumas","alguns","ali","além","ambas","ambos","ano","anos","antes","ao","aonde","aos","apenas","apoio","apontar","apos","após","aquela","aquelas","aquele","aqueles","aqui","aquilo","as","assim","através","atrás","até","aí","baixo","bastante","bem","boa","boas","bom","bons","breve","cada","caminho","catorze","cedo","cento","certamente","certeza","cima","cinco","coisa","com","como","comprido","conhecido","conselho","contra","contudo","corrente","cuja","cujas","cujo","cujos","custa","cá","da","daquela","daquelas","daquele","daqueles","dar","das","de","debaixo","dela","delas","dele","deles","demais","dentro","depois","desde","desligado","dessa","dessas","desse","desses","desta","destas","deste","destes","deve","devem","deverá","dez","dezanove","dezasseis","dezassete","dezoito","dia","diante","direita","dispoe","dispoem","diversa","diversas","diversos","diz","dizem","dizer","do","dois","dos","doze","duas","durante","dá","dão","dúvida","e","ela","elas","ele","eles","em","embora","enquanto","entao","entre","então","era","eram","essa","essas","esse","esses","esta","estado","estamos","estar","estará","estas","estava","estavam","este","esteja","estejam","estejamos","estes","esteve","estive","estivemos","estiver","estivera","estiveram","estiverem","estivermos","estivesse","estivessem","estiveste","estivestes","estivéramos","estivéssemos","estou","está","estás","estávamos","estão","eu","exemplo","falta","fará","favor","faz","fazeis","fazem","fazemos","fazer","fazes","fazia","faço","fez","fim","final","foi","fomos","for","fora","foram","forem","forma","formos","fosse","fossem","foste","fostes","fui","fôramos","fôssemos","geral","grande","grandes","grupo","ha","haja","hajam","hajamos","havemos","havia","hei","hoje","hora","horas","houve","houvemos","houver","houvera","houveram","houverei","houverem","houveremos","houveria","houveriam","houvermos","houverá","houverão","houveríamos","houvesse","houvessem","houvéramos","houvéssemos","há","hão","iniciar","inicio","ir","irá","isso","ista","iste","isto","já","lado","lhe","lhes","ligado","local","logo","longe","lugar","lá","maior","maioria","maiorias","mais","mal","mas","me","mediante","meio","menor","menos","meses","mesma","mesmas","mesmo","mesmos","meu","meus","mil","minha","minhas","momento","muito","muitos","máximo","mês","na","nada","nao","naquela","naquelas","naquele","naqueles","nas","nem","nenhuma","nessa","nessas","nesse","nesses","nesta","nestas","neste","nestes","no","noite","nome","nos","nossa","nossas","nosso","nossos","nova","novas","nove","novo","novos","num","numa","numas","nunca","nuns","não","nível","nós","número","o","obra","obrigada","obrigado","oitava","oitavo","oito","onde","ontem","onze","os","ou","outra","outras","outro","outros","para","parece","parte","partir","paucas","pegar","pela","pelas","pelo","pelos","perante","perto","pessoas","pode","podem","poder","poderá","podia","pois","ponto","pontos","por","porque","porquê","portanto","posição","possivelmente","posso","possível","pouca","pouco","poucos","povo","primeira","primeiras","primeiro","primeiros","promeiro","propios","proprio","própria","próprias","próprio","próprios","próxima","próximas","próximo","próximos","puderam","pôde","põe","põem","quais","qual","qualquer","quando","quanto","quarta","quarto","quatro","que","quem","quer","quereis","querem","queremas","queres","quero","questão","quieto","quinta","quinto","quinze","quáis","quê","relação","sabe","sabem","saber","se","segunda","segundo","sei","seis","seja","sejam","sejamos","sem","sempre","sendo","ser","serei","seremos","seria","seriam","será","serão","seríamos","sete","seu","seus","sexta","sexto","sim","sistema","sob","sobre","sois","somente","somos","sou","sua","suas","são","sétima","sétimo","só","tal","talvez","tambem","também","tanta","tantas","tanto","tarde","te","tem","temos","tempo","tendes","tenha","tenham","tenhamos","tenho","tens","tentar","tentaram","tente","tentei","ter","terceira","terceiro","terei","teremos","teria","teriam","terá","terão","teríamos","teu","teus","teve","tinha","tinham","tipo","tive","tivemos","tiver","tivera","tiveram","tiverem","tivermos","tivesse","tivessem","tiveste","tivestes","tivéramos","tivéssemos","toda","todas","todo","todos","trabalhar","trabalho","treze","três","tu","tua","tuas","tudo","tão","tém","têm","tínhamos","um","uma","umas","uns","usa","usar","vai","vais","valor","veja","vem","vens","ver","verdade","verdadeiro","vez","vezes","viagem","vindo","vinte","você","vocês","vos","vossa","vossas","vosso","vossos","vários","vão","vêm","vós","zero","à","às","área","é","éramos","és","último"])
    stop_words_dict = {"eu":eu,"ca":ca,"gl":gl,"es":es,"en":en,"pt":pt}
    all_stop_words = eu.union(ca,gl,es,en,pt)
    word_list = t.tweet.split(" ")
    processed_tweet = ""

    # if ever need to print tokens to file turn on print toggle
    if print_toggle:
        f = open("processed_tweets.txt","a",encoding="utf-8")

    # start to do preprocess    
    for word in word_list:
        if len(word) is 0:
            break
        if word[0:4] == "http":  # no webaddress
            break
        if word[0] == "#": # no "#", they are often in english
            break
        # delete repetitive characters like YAAAAAAAAHHHHHHHHHHHH or HAHAHAHAHAHAHA, may overkill words like meet
        for x in reversed(range(1,5)):
            for i in range(len(word)):
                while word.lower()[i:i+x] == word.lower()[i+x:i+(x*2)]:
                    word = word[:i+x]+word[i+(x*2):]
                    if i>=len(word):
                        break
                if i>=len(word):
                    break
        # if only want to run n-gram for stopwords,  turn it on, it's off now, bad accuracy for tweets since they are too short.
        if stop_words_toggle:
            if is_test:
                if word not in all_stop_words:
                    break
            else:
                if word not in stop_words_dict[t.language]:
                    break
        processed_tweet = processed_tweet + word + " "
    if print_toggle:
        f.write(processed_tweet+"\n")
    return processed_tweet

def executeNgram(V, gamma, n, training, testing):
    testingData = processData(testing)

    # language frequencies denoted by _f
    eu_p, ca_p, gl_p, es_p, en_p, pt_p, language_p = buildNgramModelByVocabulary(V, training, gamma, n)

    TP_counts = {"eu":0,"ca":0,"gl":0,"es":0,"en":0,"pt":0}
    real_label_counts = {"eu":0,"ca":0,"gl":0,"es":0,"en":0,"pt":0}
    prediction_counts = {"eu":0,"ca":0,"gl":0,"es":0,"en":0,"pt":0}
    total_correct = 0
    total_test_tweets = 0
    for t in testingData:
        real_label = t.language
        total_test_tweets += 1
        if V == 3:    
            stop_words_toggle = False
            print_toggle = False
            is_test = True
            t.tweet = tweet_preprocess(t, stop_words_toggle, print_toggle, is_test)

        prediction, probability = detectTweetNgram(V, t, eu_p, ca_p, gl_p, es_p, en_p, pt_p, language_p, n)
        
        real_label_counts[real_label] += 1
        prediction_counts[prediction] += 1
        if prediction == real_label:
            total_correct += 1
            TP_counts[prediction] += 1
        writeToTraceFile(t, prediction, probability, V, n, gamma)
        
    accuracy = '{:.4f}'.format(total_correct/total_test_tweets)
    precision_dict = {"eu":0,"ca":0,"gl":0,"es":0,"en":0,"pt":0}
    recall_dict = {"eu":0,"ca":0,"gl":0,"es":0,"en":0,"pt":0}
    f1_dict = {"eu":0,"ca":0,"gl":0,"es":0,"en":0,"pt":0}
    for key in TP_counts.keys():
        if prediction_counts[key] != 0:
            precision_dict[key] = TP_counts[key]/prediction_counts[key]
        else:
            precision_dict[key] = None
        if real_label_counts[key] != 0:
            recall_dict[key] = TP_counts[key]/real_label_counts[key]
        else:
            recall_dict[key] = None
        if precision_dict[key] is None and recall_dict[key] is None:
            f1_dict[key] = None
        else:
            if precision_dict[key] != 0 and recall_dict[key] != 0:
                f1_dict[key] = (2*precision_dict[key]*recall_dict[key])/(precision_dict[key]+recall_dict[key])
            else:
                f1_dict[key] = None
    macro_f1 = 0
    weighted_f1 = 0
    for key in f1_dict.keys():
        if f1_dict[key] is None:
            macro_f1 = None
            weighted_f1 = None
            break
        else:
            macro_f1 = macro_f1 + f1_dict[key]/len(f1_dict)
            weighted_f1 = weighted_f1 + f1_dict[key]*real_label_counts[key]/total_test_tweets
    
    if macro_f1 is not None:        
        macro_f1 = '{:.4f}'.format(macro_f1)
    if weighted_f1 is not None:
        weighted_f1 = '{:.4f}'.format(weighted_f1)
    for k in precision_dict.keys():
        if precision_dict[k] is not None:
            precision_dict[k] = '{:.4f}'.format(precision_dict[k])
    for k in recall_dict.keys():
        if recall_dict[k] is not None:
            recall_dict[k] = '{:.4f}'.format(recall_dict[k])
    for k in f1_dict.keys():
        if f1_dict[k] is not None:
            f1_dict[k] = '{:.4f}'.format(f1_dict[k])

    with open("eval_" + str(V) + "_" + str(n) + "_" + str(gamma) + ".txt", "w", encoding="utf-8",) as f:
        f.write(
            str(accuracy) + "\n"
            + str(precision_dict["eu"]) + "  " + str(precision_dict["ca"]) + "  " + str(precision_dict["gl"]) + "  " + str(precision_dict["es"]) + "  " + str(precision_dict["en"]) + "  " + str(
                precision_dict["pt"]) + "\n"
            + str(recall_dict["eu"]) + "  " + str(recall_dict["ca"]) + "  " + str(recall_dict["gl"]) + "  " + str(recall_dict["es"]) + "  " + str(recall_dict["en"]) + "  " + str(
                recall_dict["pt"]) + "\n"
            + str(f1_dict["eu"]) + "  " + str(f1_dict["ca"]) + "  " + str(f1_dict["gl"]) + "  " + str(f1_dict["es"]) + "  " + str(f1_dict["en"]) + "  " + str(
                f1_dict["pt"]) + "\n"
            + str(macro_f1) + "  " + str(weighted_f1)
        )

def buildNgramModelByVocabulary(V, twitter_posts, gamma, n):

    if V == 0:
        # get dictionary that has all the possible combinations of letters. example when n=3: {aaa:0,aab:0,aac:0 ... ...,zzx:0,zzy:0,zzz:0}
        eu_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)]))  # Basque
        ca_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)]))  # Catalan
        gl_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)]))  # Galician
        es_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)]))  # Spanish
        en_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)]))  # English
        pt_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_lowercase, repeat=n)])) # Portuguese
        # print(len(pt_n_gram_dict))
    elif V == 1 or V == 3:
        # get dictionary that has all the possible combinations of letters. example when n=3: {aaa:0,aab:0,aac:0 ... ...,zzx:0,zzy:0,zzz:0}
        eu_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # Basque
        ca_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # Catalan
        gl_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # Galician
        es_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # Spanish
        en_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # English
        pt_n_gram_dict = dict(map(lambda x: (x,0), [''.join(x) for x in product(string.ascii_letters, repeat=n)]))  # Portuguese
        # print(len(pt_n_gram_dict))
    elif V == 2:
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
            buildNgramLanguageModel(eu_n_gram_dict, twitter_post, V, n) # {"abs": 1, "axs": 1, ...}
        if twitter_post.language == "ca":
            language_probability['ca'] = language_probability['ca'] + 1
            buildNgramLanguageModel(ca_n_gram_dict, twitter_post, V, n)
        if twitter_post.language == "gl":
            language_probability['gl'] = language_probability['gl'] + 1
            buildNgramLanguageModel(gl_n_gram_dict, twitter_post, V, n)
        if twitter_post.language == "es":
            language_probability['es'] = language_probability['es'] + 1
            buildNgramLanguageModel(es_n_gram_dict, twitter_post, V, n)
        if twitter_post.language == "en":
            language_probability['en'] = language_probability['en'] + 1
            buildNgramLanguageModel(en_n_gram_dict, twitter_post, V, n)
        if twitter_post.language == "pt":
            language_probability['pt'] = language_probability['pt'] + 1
            buildNgramLanguageModel(pt_n_gram_dict, twitter_post, V, n) 
    for key, value in language_probability.items():
        language_probability[key] = math.log(value / total_tweets) # list: [log(probability of a language showing up)]
    if V == 2:
        eu_n_gram_dict["NOT-APPEAR"] = 0  # Basque       {"abs": 1, "axs": 1, ..., "NOT-APPEAR": 0}
        ca_n_gram_dict["NOT-APPEAR"] = 0   # Catalan
        gl_n_gram_dict["NOT-APPEAR"] = 0   # Galician
        es_n_gram_dict["NOT-APPEAR"] = 0   # Spanish
        en_n_gram_dict["NOT-APPEAR"] = 0   # English
        pt_n_gram_dict["NOT-APPEAR"] = 0   # Portuguese

    eu_ngram_probability = NgramConditionalProbability(eu_n_gram_dict, gamma) # smoothing if needed, {"abs": (1+0.1)/N+V*0.1, "axs": (1+0.1)/N+V*0.1, ..., "NOT-APPEAR": (0+0.1)/N+V*0.1}
    ca_ngram_probability = NgramConditionalProbability(ca_n_gram_dict, gamma)
    gl_ngram_probability = NgramConditionalProbability(gl_n_gram_dict, gamma)
    es_ngram_probability = NgramConditionalProbability(es_n_gram_dict, gamma)
    en_ngram_probability = NgramConditionalProbability(en_n_gram_dict, gamma)
    pt_ngram_probability = NgramConditionalProbability(pt_n_gram_dict, gamma)
    return (eu_ngram_probability, ca_ngram_probability, gl_ngram_probability, es_ngram_probability, en_ngram_probability, pt_ngram_probability, language_probability)

def buildNgramLanguageModel(ngram_dict, twitter_post, V, n):

    if V == 0: 
        tweet = twitter_post.tweet.lower()
        for i in range(len(tweet)-n):
            ngram = tweet[i:i+n] # get an ngram, "a", "aa", or "aaa"
            if ngram in ngram_dict.keys():
                ngram_dict[ngram] += 1
            


    if V == 1 or V == 3:
        tweet = twitter_post.tweet
        for i in range(len(tweet)-n):
            ngram = tweet[i:i+n]
            if ngram in ngram_dict.keys():
                ngram_dict[ngram] += 1


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
    p = 0
    if V == 0:
        tweet = twitterPost.tweet.lower()
        for i in range(len(tweet)-n):
            ngram = tweet[i:i+n]
            if ngram in target_language_p.keys():
                if target_language_p[ngram] == 0:
                    p += -math.inf
                else:
                    p += math.log(target_language_p[ngram])

        
    if V == 1 or V == 3:
        tweet = twitterPost.tweet
        for i in range(len(tweet)-n):
            ngram = tweet[i:i+n]
            if ngram in target_language_p.keys():
                if target_language_p[ngram] == 0:
                    p += -math.inf
                else:
                    p += math.log(target_language_p[ngram])
    if V == 2:
        tweet = twitterPost.tweet
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


# following borrowed function is used to circumvent encoding errors when printing dictionaries with isalpha allowed characters
# https://stackoverflow.com/questions/14630288/unicodeencodeerror-charmap-codec-cant-encode-character-maps-to-undefined
def uprint(*objects, sep=" ", end="\n", file=sys.stdout):
    enc = file.encoding
    if enc == "UTF-8":
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors="backslashreplace").decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)


def main():

    V=0
    n=1
    gamma = 0
    print("V=%d, n=%d, gamma=%f" % (V,n,gamma))
    training = os.path.join(sys.path[0], "training-tweets.txt")
    testing = os.path.join(sys.path[0], "test-tweets-given.txt")
    executeNaiveBayesClassification(V, n, gamma, training, testing)

    V=1
    n=2
    gamma = 0.5
    print("V=%d, n=%d, gamma=%f" % (V,n,gamma))
    training = os.path.join(sys.path[0], "training-tweets.txt")
    testing = os.path.join(sys.path[0], "test-tweets-given.txt")
    executeNaiveBayesClassification(V, n, gamma, training, testing)

    V=1
    n=3
    gamma = 1
    print("V=%d, n=%d, gamma=%f" % (V,n,gamma))
    training = os.path.join(sys.path[0], "training-tweets.txt")
    testing = os.path.join(sys.path[0], "test-tweets-given.txt")
    executeNaiveBayesClassification(V, n, gamma, training, testing)

    V=2
    n=2
    gamma = 0.3
    print("V=%d, n=%d, gamma=%f" % (V,n,gamma))
    training = os.path.join(sys.path[0], "training-tweets.txt")
    testing = os.path.join(sys.path[0], "test-tweets-given.txt")
    executeNaiveBayesClassification(V, n, gamma, training, testing)
    
    # BYOM
    V=3
    n=2
    gamma = 0.1
    print("BYOM")
    # print("V=%d, n=%d, gamma=%f" % (V,n,gamma))
    training = os.path.join(sys.path[0], "training-tweets.txt")
    testing = os.path.join(sys.path[0], "test-tweets-given.txt")
    executeNaiveBayesClassification(V, n, gamma, training, testing)



if __name__ == "__main__":
    main()




