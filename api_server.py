from flask import (
    Flask,
    render_template,
    url_for,
    request
)

import connexion

import os,sys
import subprocess
import logging
import json

import handler

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
####################################################################################################
                                        FUNCTIONS
####################################################################################################
"""




"""
####################################################################################################
                                        API SERVER
####################################################################################################
"""
"""
---------------------------------------------------------------------------------------------------
Create the application instance
---------------------------------------------------------------------------------------------------
"""
app = Flask(__name__, template_folder="templates")

"""
---------------------------------------------------------------------------------------------------
Create main URL route "/"
---------------------------------------------------------------------------------------------------
"""
@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')

"""
---------------------------------------------------------------------------------------------------
Create a URL route for form "/form"
---------------------------------------------------------------------------------------------------
"""
@app.route('/form',methods=['GET'])
def form():
    return render_template('form.html')


"""
---------------------------------------------------------------------------------------------------
Create a URL route for "/prediction"
---------------------------------------------------------------------------------------------------
"""
@app.route('/predict',methods=['POST'])
def prediction():

    request_form = request.form
    post = request_form['post']
    post_str = str(post)

    tags_supervised_str = handler.supervised_prediction(post_str)
    tags_supervised = []
    for t in tags_supervised_str.split():
        tags_supervised.append(t)

    tags_unsupervised_str = handler.unsupervised_prediction(post_str)
    tags_unsupervised = []
    for t in tags_unsupervised_str.split():
        tags_unsupervised.append(t)

        
    return render_template("result.html", post=post_str, result_supervised=tags_supervised, len_supervised=len(tags_supervised), result_unsupervised=tags_unsupervised, len_unsupervised=len(tags_unsupervised))


"""
---------------------------------------------------------------------------------------------------
Run the application
---------------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    logger.info("ok")

    app.run(host='0.0.0.0',port=5000,debug=False)

    