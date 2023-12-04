import weka.core.jvm as jwm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
from weka.core.classes import Random
import weka.plot.graph as graph


jwm.start(packages=True)
#convert=input("Aukeratu convert-a: ")
#filter= input("Aukeratu filterr-a: ")
convert="Edge"
filter="PHOGFilter"
def data_loader(convert, filter):
    data_dir = "./DMDatuak/" + convert + "/"
    loader = Loader(classname="weka.core.converters.ArffLoader")
    print(data_dir + filter + "_" + convert + ".arff")
    data = loader.load_file(data_dir + filter + "_" + convert + ".arff")
    data.class_is_last()
    return data

data = data_loader(convert, filter)
print(data)

remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
remove.inputformat(data)
filtered_data = remove.filter(data)

data.class_is_last()	
classifier = Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "1.0", "-L", "0.001", "-P", "1.0E-12", "-N", "0", "-V", "-1", "-W", "1"])
evaluation = Evaluation (filtered_data)
evaluation.crossvalidate_model (classifier, filtered_data, 10, Random(42)) 
print (evaluation.summary())
print ("pctCorrect: "+ str(evaluation.percent_correct))
print ("incorrect: " + str (evaluation.incorrect))

#classifier = Classifier(classname="weka.filters.supervised.attribute.AttributeSelection -S "weka.attributeSelection.Ranker -N 50"  -E "weka.attributeSelection.ReliefFAttributeEval -M -1 -D 1 -K 10" -i $1.arff -o $1FSS.arff -r $2.arff -s $2FSS.arff -c last -b")

jwm.stop()
