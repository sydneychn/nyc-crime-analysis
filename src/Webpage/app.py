from flask import Flask, render_template, request
import pandas as pd
from Algorithms.clean import *
from Algorithms.KNN import *
from Algorithms.One_Class_Svm import *
from Algorithms.Linear_Reg import *
from Algorithms.SVM import *
from Algorithms.Kmeans import *
from Algorithms.chi_square import *
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    algorithm = None
    chi_result = None
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['file']
        if file:
            # Read CSV file
            data = pd.read_csv(file)
            
            # Clean CSV file
            data = clean(data)
            
            # Choose algorithm based on selection
            algorithm = request.form['algorithm']
            if algorithm == 'knn':
                result = knn(data)
            elif algorithm == 'one_class_svm':
               result = one_class_svm(data)
            elif algorithm == 'linear_reg':
                result = linear_reg(data)
            elif algorithm == 'svm':
                result = testSVMWithCSV(data)
            elif algorithm == 'kmeans':
                result = testKMeansWithCSV(data)
            elif algorithm == 'chi_square':
                chi_result = chi_square(data)
    return render_template('index.html', result=result, algorithm=algorithm, chi_result=chi_result) 



if __name__ == '__main__':
    app.run(debug=True)

