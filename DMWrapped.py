import weka.core.jvm as jwm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
from weka.core.classes import Random

jwm.start()
def data_loader(convert, filter):
    data_dir = "/users/1059715/Desktop/DMWrapped/DMDatuak/" + convert
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(data_dir + filter + "_" + convert + ".arff")
    data.class_is_last()


jwm.stop()
