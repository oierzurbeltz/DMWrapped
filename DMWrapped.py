import weka.core.jvm as jwm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
from weka.core.classes import Random
import weka.plot.graph as graph


jwm.start()
convert=input("Aukeratu convert-a: ")
filter= input("Aukeratu filterr-a: ")
def data_loader(convert, filter):
    data_dir = "./DMDatuak/" + convert + "/"
    loader = Loader(classname="weka.core.converters.ArffLoader")
    print(data_dir + filter + "_" + convert + ".arff")
    data = loader.load_file(data_dir + filter + "_" + convert + ".arff")
    data.class_is_last()
    return data

data = data_loader(convert, filter)
print(data)

cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
cls.build_classifier(data)

print(cls)
graph.plot_dot_graph(cls.graph)


jwm.stop()
