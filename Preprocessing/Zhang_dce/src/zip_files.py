import os
import zipfile

os.chdir(os.path.dirname(__file__))
os.chdir(os.pardir)

zip_file = zipfile.ZipFile('code_data.zip', 'w')

files = [
    # adult data
    'data/Adult.csv',
    'data/Dutch.csv',
    'data/test/test.csv',
    # src
    'src/main.py',
    'src/main_blockset.py',
    'src/zip_files.py',
    'src/RemoveAlgorithms/Basic.py',
    'src/RemoveAlgorithms/BinarizeAdult.py',
    'src/RemoveAlgorithms/Blackbox.py',
    'src/RemoveAlgorithms/Detection.py',
    'src/RemoveAlgorithms/HandlingConditionalDiscrimination.py',
    'src/RemoveAlgorithms/MData.py',
    'src/RemoveAlgorithms/MGraph.py',
    'src/RemoveAlgorithms/Naive.py',
    'src/RemoveAlgorithms/Utility.py',
    'src/RemoveAlgorithms/__init__.py',
    'src/RemoveAlgorithms/blackboxrepairers/__init__.py',
    'src/RemoveAlgorithms/blackboxrepairers/AbstractRepairer.py',
    'src/RemoveAlgorithms/blackboxrepairers/binning/__init__.py',
    'src/RemoveAlgorithms/blackboxrepairers/binning/Binner.py',
    'src/RemoveAlgorithms/blackboxrepairers/binning/BinSizes.py',
    'src/RemoveAlgorithms/blackboxrepairers/calculators.py',
    'src/RemoveAlgorithms/blackboxrepairers/CategoricalFeature.py',
    'src/RemoveAlgorithms/blackboxrepairers/CategoricRepairer.py',
    'src/RemoveAlgorithms/blackboxrepairers/GeneralRepairer.py',
    'src/RemoveAlgorithms/blackboxrepairers/NumericRepairer.py',
    'src/RemoveAlgorithms/blackboxrepairers/SparseList.py',
    'README.MD'
]
for f in files:
    zip_file.write(f, compress_type=zipfile.ZIP_DEFLATED)

zip_file.close()
