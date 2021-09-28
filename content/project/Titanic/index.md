---
title: Kaggle Titanic Survival Prediction  
summary: An example of using the in-built project page.
tags:
- Machine Learning
date: "2016-04-27T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: "https://jordanjames.netlify.app/project/proj/titanic_predictions.html"

---
<html>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><h1 style="text-align: center;">Kaggle Titanic Survival Predictions</h1></p>
<p><h3>Part One: Titanic Tutorial - Random Forest Classifier</h3></p>
<p>This blog post looks at the Kaggle challenge, Titanic - Machine Learning from Disaster, found at https://www.kaggle.com/c/titanic/overview. Kaggle provides a tutorial for this challenge as a way to help users familiarize themselves with the site. This tutorial can be found at https://www.kaggle.com/alexisbcook/titanic-tutorial/notebook. In part one of this blog post we will follow the tutorial and produce a random forest classifier to predict the survival rate of passengers. All code from part one is referenced from the tutorial at https://www.kaggle.com/alexisbcook/titanic-tutorial/notebook</p>
<p>Pandas is the package that will be used to handle the data thus numpy is also required as a dependency. Pandas allows us to easily read in the data with the test and train csv files provided on the Kaggle challenge site.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span> <span class="c1"># linear algebra</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span> <span class="c1"># data processing, CSV file I/O (e.g. pd.read_csv)</span>
</pre></div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The data is given by Kaggle in two seperate csv files. One is the training data labelled "train.csv" and the other is the testing data labelled "test.csv". The two files are very similar but the key difference is that the training data has the attribute Survived whereas the test data does not. This attribute is the ground truth value and is only provided for the training data. This is the attribute that we will be trying to predict for the test data and the resulting predictions will be verified by Kaggle upon submission. </p>
<p>To begin we read both of the csv files in and display the first 5 rows of each to get an idea of what the data looks like. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">train_data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;train.csv&quot;</span><span class="p">)</span>
<span class="n">train_data</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">




<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">test_data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;test.csv&quot;</span><span class="p">)</span>
<span class="n">test_data</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">




<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Observing the rows above we see that the data has ten attributes and a unique passenger ID. The attributes are given on Kaggle in the Data Dictionary, this dictionary is depicted below.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src="data:image/png;base64,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" alt="DataDictionary.PNG"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now that we understand what the data is and how it is structured we can further analyze the attributes. We know we want to predict the Survived attribute so we want to see if we can find any patterns with this attribute in the training dataset. The blocks of code below partition the data based on the sex of the passenger and calculates the survival rate for each value of the attribute sex.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">women</span> <span class="o">=</span> <span class="n">train_data</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">train_data</span><span class="o">.</span><span class="n">Sex</span> <span class="o">==</span> <span class="s1">&#39;female&#39;</span><span class="p">][</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span>
<span class="n">rate_women</span> <span class="o">=</span> <span class="nb">sum</span><span class="p">(</span><span class="n">women</span><span class="p">)</span><span class="o">/</span><span class="nb">len</span><span class="p">(</span><span class="n">women</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">% o</span><span class="s2">f women who survived:&quot;</span><span class="p">,</span> <span class="n">rate_women</span><span class="p">)</span>
</pre></div>

</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">



<div class="output_subarea output_stream output_stdout output_text">
<pre>% of women who survived: 0.7420382165605095
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">men</span> <span class="o">=</span> <span class="n">train_data</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">train_data</span><span class="o">.</span><span class="n">Sex</span> <span class="o">==</span> <span class="s1">&#39;male&#39;</span><span class="p">][</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span>
<span class="n">rate_men</span> <span class="o">=</span> <span class="nb">sum</span><span class="p">(</span><span class="n">men</span><span class="p">)</span><span class="o">/</span><span class="nb">len</span><span class="p">(</span><span class="n">men</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">% o</span><span class="s2">f men who survived:&quot;</span><span class="p">,</span> <span class="n">rate_men</span><span class="p">)</span>
</pre></div>

</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">



<div class="output_subarea output_stream output_stdout output_text">
<pre>% of men who survived: 0.18890814558058924
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The quick analysis of the survival rate of passengers based on sex shows a clear pattern. We can see that it is much more likely that a female passenger survived than a male passenger. This is a good indicator that sex is an important attribute when it comes to predicting the passengers survival rate and should be considered as a feature in our model.  </p>
<p>The following block of code uses the python package sklearn to import a Random Forest Classifier that will be used as our model to predict the survival rate of passengers. The model we build below uses as features the four attributes Pclass, Sex, SibSp, and Parch. A Random Forest Classifier is essentially just multiple decision trees using the attributes given to decide whether or not a passenger survived and its output is the maximum of the output of the trees (mostly survived/mostly not survived). Random Forest Classifiers typically suffer from the problem of overfitting the training data however this is not a problem if the test data is very similar to the training data. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>

<span class="n">y</span> <span class="o">=</span> <span class="n">train_data</span><span class="p">[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span>

<span class="n">features</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Pclass&quot;</span><span class="p">,</span> <span class="s2">&quot;Sex&quot;</span><span class="p">,</span> <span class="s2">&quot;SibSp&quot;</span><span class="p">,</span> <span class="s2">&quot;Parch&quot;</span><span class="p">]</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">train_data</span><span class="p">[</span><span class="n">features</span><span class="p">])</span>
<span class="n">X_test</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">test_data</span><span class="p">[</span><span class="n">features</span><span class="p">])</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span> <span class="n">max_depth</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
<span class="n">predictions</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>

<span class="n">output</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;PassengerId&#39;</span><span class="p">:</span> <span class="n">test_data</span><span class="o">.</span><span class="n">PassengerId</span><span class="p">,</span> <span class="s1">&#39;Survived&#39;</span><span class="p">:</span> <span class="n">predictions</span><span class="p">})</span>
<span class="n">output</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s1">&#39;submission.csv&#39;</span><span class="p">,</span> <span class="n">index</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Your submission was successfully saved!&quot;</span><span class="p">)</span>

<span class="c1">#All code above this line is referenced from the tutorial at https://www.kaggle.com/alexisbcook/titanic-tutorial/notebook</span>
</pre></div>

</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">



<div class="output_subarea output_stream output_stdout output_text">
<pre>Your submission was successfully saved!
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The submission is saved to a csv file named submission.csv and is ready for checking on Kaggle. It can be seen below that the random forest classifier achieves an accuracy score of 77.511% on the test data. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA8UAAADkCAYAAACxKoZiAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAADNbSURBVHhe7Z15tCVFnef7jzlnzpk5Z0ZFpV1LFBTEBVnskXY5NqNtdyPS7gja2NAw3fTpURnbrV167FF7ulsdmmaxLAqKpSgsdmSTrVhLZCuEAoFiKSiKwqIoFoWS9sS8T777exUvKu/N+6reffXuzc/nnO95NzMiIyMzIyLjmxGZ73eSiIiIiIiISEvRFIuIiIiIiEhr0RSLiIiIiIhIa9EUi4iIiIiISGvRFIuIiIiIiEhr0RSLiIiIiIhIa9EUi4iIiIiISGvpyxQ/u2FDWr36kXTvffcrpZRSSimllFKzXnhYvGwTjaaYREjw8cfXp+eee04ppZRSSimllJr1wsPiZZuMcaMpxl1riJVSSimllFJKDZvwsnjaXjSaYpx1XeJKKaWUUkoppdRsF562F5pipZRSSimllFIjK02xUkoppZRSSqnWSlOslFJKKaWUUqq10hQrpZRSSimllGqtNMVKKaWUUkoppVorTbFSSimllFJKqdZKU6yUUkoppZRSqrXSFCullFJKKaWUaq00xUoppZRSSimlWitNsVJKKaWUUkqp1kpTrJRSSimllFKqtdIUK6WUUkoppZRqrTTFSimllFJKKaVaK02xUkoppZRSSqnWSlOslFJKKaWUUqq1GgpT/NRTT6XHH388bdiwYZOwX/3qV1XYM888s0mYGn2tX7++Kh91YZur2267Lf3sZz+rDZuK7r///nTFFVfUhimllFJKKaVmh4bCFK9YsSIdffTR6brrrpu0HpN8wgknpJNOOmnS+lETJm316tW1YbNZTz/9dJX3devW1YZPh7j25513Xm3Y5uqUU05Jc+fOrQ2bis4///yq3PLQpi5cKaWUUkoptfU1FKYYYXyOOeaY9OSTT06sYzQP07Fy5cpJcUdNHOM111xTGzabtWbNmirvd9xxR234dGgQpphZB8xAqAubinhoM92j2EoppZRSSqnp1dCYYswwpvjiiy+etMxoXMRhKu0ll1yS5s+fny644IL06KOPToQxWse6hx56aGIdv1kXI3n8Xr58efrpT39apZFvH7rxxhvTlVdeWU2N/dGPfpTOPPPMtHbt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j22qNA+ct0va729fUhnxNx6+y3j8Is0qD3u8ML3178byMGZsDzpi+7TrnrlRfn1629vGlvfdPu03b9d0yNi6cXP/gvTmw3ZKnzpp93Tw3J3SO6op1ZONslJKKaWUUkoNuwbNaJriMR36g9el//ae51dfi56D9nh+ev3+r00HLMzj7Zo+csi26dVjRriKs9vz004H7pg+HR+w2kxTvNOnX5N+n69OT6S5Uzoo4m+SZpmH56UdPvCatF+Wz/0//7tphyp82/Te41i3e/rUl16WXsuXrzvbbLfXK9O+8/MRZqWUUkoppZQafg2akTXFE1q4azqwM8JaG44W7ZY+Pe8t6eCJrzlvrvKpzrung+fvkg6aZMJ7qMrDFOJXmvwvppRSSimllFJq1DRoRt8Uz6jK93+VUkoppZRSSm2JBo2meFq1U9rzndukXT735powpZRSSimllFJT1aDRFCullFJKKaWUmrUaNJpipZRSSimllFKzVoNGU6yUUkoppZRSatZq0GiKlVJKKaWUUkrNWg0aTbFSSimllFJKqVmrQaMpVkoppZRSSik1azVoNMVKKaWUUkoppWatBs1QmGIRERERERGRQaApFhERERERkdaiKRYREREREZHWoikWERERERGR1qIpFhERERERkdaiKRYREREREZHWoikWERERERGR1qIpFhERERERkdaiKRYREREREZHWoikWERERERGR1qIpFhERERERkdaiKRYREREREZHWoikWERERERGR1jK0pnjlk8+kp3/z750lGSaOvOnBtPrpDZ2lTbl97dPppNtXd5aGlxWP/3pay+gjY+cMDZLpqFcc94Z//21naetiOyEiIiIiTQytKX7VsVenH966qrM0O/i3MbP32DO/6SxJHfeu/3X6D/98afo/193XWbPpedvv3J+n//r/rugsDQdLH34iXXDv2s7SOL/zfy+Z1jL63tNuqjRIpqNecdw3PPJkZ2nrMhvbCRERERGZXWiKp5HZZAaGiVE4b//z0l9sYlg1xVsfTbGIiIiINDESpviE2x5Oh150R/U7+Oa191brgdHJ959+S3rgiWfSf190U3rdD69NR9/8UDXFEzPzymOuTgdfuHzSNEvi/2Ldr6r1LzpySfr4OT+vtq8j0scMvGvhDdW+A9IkDfbxB6femG5e090sMK34a1ev6CyNQx5Ie/2zz1XL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Kx6v/0DVdEOnOzrT7DM67JsLacTIVxOYYjr1+ReXc+I8bGme6sgNR9P5bsoHxzud+XxkzHii6aIp/zNF5KPf8lHCOSk/Zhase+a5rmEB4cTrxpbkr+n89kq3n7zX0XQ8IiIiIiI5Q2WKZ5LcFM8kfHGXfw0U/x5opuk2CiciIiIiIjKKDIUpxqTNpOBFRy5Jpyx/pDZ8UHroqWer/TJFlem4dXEGLf4VE3ngb124UqMgEREREZHAkWIRERERERFpLZpiERERERERaS2aYhEREREREWktmmIRERERERFpLZpiERERERERaS2aYhEREREREWktmmIRERERERFpLZpiERERERERaS2aYhEREREREWktmmIRERERERFpLZpiERERERERaS2aYhEREREREWktmmIRERERERFpLZpiERERERERaS2aYhEREREREWktmmIRERERERFpLZpiERERERERaS2aYhEREREREWktmmIRERERERFpLZpiERERERERaS2aYhEREREREWktmmIRERERERFpLZpiERERERERaS1bbIofeWRNeuKJJztLIiIiIiIiIsMBXhZP24tGU/zshg2Vs9YYi4iIiIiIyLCAh8XL4ml70WiKgURw1ySolFJKKaWUUkrNduFhmwwx9GWKRUREREREREYRTbGIiIiIiIi0Fk2xiIiIiIiItBZNsYiIiIiIiLQWTbGIiIiIiIi0Fk2xiIiIiIiItBZNsYiIiIiIiLQWTbGIiIiIiIi0Fk2xiIiIiIiItBZNsYiIiIiIiLQWTbGIiIiIiIi0Fk2xiIiIiIiItBZNsYiIiIiIiLQWTbGIiIiIiIi0lJT+P4Z84K9bpmjUAAAAAElFTkSuQmCC" alt="firstsubmit.PNG"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><h1 style="text-align: center;">Contribution</h1></p>
<p><h3>Part Two: Refining the Random Forest Classifier and K-Nearest Neighbors Classifier</h3></p>
<p>This portion of the blog focuses on improving the accuracy score achieved above. The first way attempted to achieve a higher accuracy score was to refine the current random forest classifier we already have. In order to do this we must find the best parameters for max depth of the trees and the number of estimators, or trees to be built. Since we are only looking into two parameters and there are only five values being considered for each, we can exhaustively search for the best parameters using a grid search with k-fold cross validation provided by sklearn. Cross validation is a method used to split the training data into k folds and treat one of these folds as test data during the training process in order to prevent overfitting the data, which is typically the main problem for a random forest classifier. The block of code below performs the grid search and outputs the best parameters found. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">GridSearchCV</span>

<span class="n">rf</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">random_state</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="c1">#establishing parameters for a grid search to find the best maximum depth and n estimators</span>
<span class="n">grid</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;max_depth&#39;</span><span class="p">:</span> <span class="p">[</span><span class="mi">5</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">15</span><span class="p">,</span> <span class="mi">20</span><span class="p">,</span> <span class="mi">25</span><span class="p">],</span>
    <span class="s1">&#39;n_estimators&#39;</span><span class="p">:</span> <span class="p">[</span><span class="mi">100</span><span class="p">,</span> <span class="mi">200</span><span class="p">,</span> <span class="mi">300</span><span class="p">,</span> <span class="mi">400</span><span class="p">,</span> <span class="mi">500</span><span class="p">]</span>
<span class="p">}</span>

<span class="n">grid_search</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">estimator</span><span class="o">=</span><span class="n">rf</span><span class="p">,</span> <span class="n">param_grid</span><span class="o">=</span><span class="n">grid</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>
<span class="n">grid_search</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">grid_search</span><span class="o">.</span><span class="n">best_params_</span><span class="p">)</span>
</pre></div>

</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">



<div class="output_subarea output_stream output_stdout output_text">
<pre>{&#39;max_depth&#39;: 5, &#39;n_estimators&#39;: 200}
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The best max depth was found to be 5, the previously used value, however the best number of estimators was found to be 200. Using the new parameters we can create a new random forest classifier and generate our predictions as we did in the tutorial. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#This code cell is referenced from the tutorial at https://www.kaggle.com/alexisbcook/titanic-tutorial/notebook</span>
<span class="n">improved_model</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="n">grid_search</span><span class="o">.</span><span class="n">best_params_</span><span class="p">[</span><span class="s1">&#39;n_estimators&#39;</span><span class="p">],</span> <span class="n">max_depth</span><span class="o">=</span><span class="n">grid_search</span><span class="o">.</span><span class="n">best_params_</span><span class="p">[</span><span class="s1">&#39;max_depth&#39;</span><span class="p">],</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">improved_model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
<span class="n">improved_predictions</span> <span class="o">=</span> <span class="n">improved_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>

<span class="n">output</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;PassengerId&#39;</span><span class="p">:</span> <span class="n">test_data</span><span class="o">.</span><span class="n">PassengerId</span><span class="p">,</span> <span class="s1">&#39;Survived&#39;</span><span class="p">:</span> <span class="n">improved_predictions</span><span class="p">})</span>
<span class="n">output</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s1">&#39;improved_submission.csv&#39;</span><span class="p">,</span> <span class="n">index</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Your submission was successfully saved!&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">



<div class="output_subarea output_stream output_stdout output_text">
<pre>Your submission was successfully saved!
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Although our model should be better than the original, it performed with the same accuracy as the original at 77.511%. The two models behave very similarly and with the small amount of data to test on, no improvement can be seen. </p>
<p>The next attempt to increase the accuracy is to use a completely different classifier than the random forest. For this the K-Nearest Neighbors Classifier was chosen. The KNN classifier determines the output of an object by the plurality vote of its neighbors and is assigned to the class that is most common among the k nearest neighbors. The classifier is already implemented in the package sklearn and can be used similar to the random forest classifier. The value used for k neighbors can be found using a grid search similar to the search used above for max depth and number of estimators. The code block below uses a grid search to find the best value for k. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="k">import</span> <span class="n">KNeighborsClassifier</span>
<span class="n">knn</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">()</span>

<span class="c1">#establishing parameters for a grid search to find the best K neighbors</span>
<span class="n">grid</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;n_neighbors&#39;</span><span class="p">:</span> <span class="p">[</span><span class="mi">5</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">15</span><span class="p">,</span> <span class="mi">20</span><span class="p">,</span> <span class="mi">25</span><span class="p">]</span>
<span class="p">}</span>
<span class="n">grid_search</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">estimator</span><span class="o">=</span><span class="n">knn</span><span class="p">,</span> <span class="n">param_grid</span><span class="o">=</span><span class="n">grid</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>
<span class="n">grid_search</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Best K:&quot;</span><span class="p">,</span> <span class="n">grid_search</span><span class="o">.</span><span class="n">best_params_</span><span class="p">[</span><span class="s1">&#39;n_neighbors&#39;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Best Score:&quot;</span><span class="p">,</span> <span class="n">grid_search</span><span class="o">.</span><span class="n">best_score_</span><span class="p">)</span>
</pre></div>

</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">



<div class="output_subarea output_stream output_stdout output_text">
<pre>Best K: 25
Best Score: 0.791256041679744
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The best value for k was found to be 25. The KNN classifier is then initialized with this value and fit to the training data. The classifier is then used to predict the survival of passengers from the test set and saved to the file submission_knn_classifier.csv. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">knn_classifier</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">(</span><span class="n">n_neighbors</span><span class="o">=</span><span class="mi">25</span><span class="p">)</span>
<span class="c1">#fit the classifier on the training dataset</span>
<span class="n">knn_classifier</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
<span class="n">knn_predictions</span> <span class="o">=</span> <span class="n">knn_classifier</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>

<span class="n">knn_output</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;PassengerId&#39;</span><span class="p">:</span> <span class="n">test_data</span><span class="o">.</span><span class="n">PassengerId</span><span class="p">,</span> <span class="s1">&#39;Survived&#39;</span><span class="p">:</span> <span class="n">knn_predictions</span><span class="p">})</span>
<span class="n">knn_output</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s1">&#39;submission_knn_classifier5.csv&#39;</span><span class="p">,</span> <span class="n">index</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Your submission was successfully saved!&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">



<div class="output_subarea output_stream output_stdout output_text">
<pre>Your submission was successfully saved!
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The output file is checked via Kaggle and the result is shown below </p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA8IAAADeCAYAAAD2F7nQAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAADdUSURBVHhe7Z15tC1VfefzR6/Va3Wv1e1MHBEFFXFABtMSoyuhNSZRQnAKggYDgU7ISqu0icYxbVpNJw5NCIP4ZHrwAEEmZZLpAQIigyCTzPMkM8ggcVXfT93zu/zufnWqzr3vnvvOvfX5rPVd91Ttql279lT7W3ufc3+jEhEREREREekRGmERERERERHpFRphERERERER6RUaYREREREREekVGmERERERERHpFRphERERERER6RWtRvjJp56q7rrr7urGm25WSimllFJKKaUmXnhYvGwbQ40wJxLJgw8+VD399NNKKaWUUkoppdTECw+Ll20zw0ONMC5aE6yUUkoppZRSaqkJL4unHcZQI4yDbopQKaWUUkoppZSadOFph6ERVkoppZRSSim17KQRVkoppZRSSinVK2mElVJKKaWUUkr1ShphpZRSSimllFK9kkZYKaWUUkoppVSvpBFWSimllFJKKdUraYSVUkoppZRSSvVKGmGllFJKKaWUUr2SRlgppZRSSimlVK+kEVZKKaWUUkop1StphJVSSimllFJK9UoaYaWUUkoppZRSvZJGWCmllFJKKaVUr6QRVkoppZRSSinVK2mElVJKKaWUUkr1ShNrhB999NHqwQcfrJ566qk1wn75y1/WYU888cQaYWr566GHHqrrR1PYfHXFFVdUP/nJTxrD5qKbb765OuussxrDlFJKKaWUUpOhiTXCN9xwQ7XPPvtU559//qz9GOODDjqoWrly5az9y00Ys7vuuqsxbJL12GOP1Wl/4IEHGsMXQpT997///caw+eqwww6r9t9//8awuejEE0+s6y0vaprClVJKKaWUUuteE2uEEWZn3333rR555JGZfczaYTRuvfXWWccuN3GPP/rRjxrDJln33HNPnfarr766MXwhNA4jzOoCVho0hc1FvKhZ6NlqpZRSSiml1MJqoo0wBhgjfOqpp87aZtYtjmGZ7GmnnVYdcMAB1UknnVTde++9M2HMyrHv9ttvn9nHZ/bFjB2fr7rqqurHP/5xHUc+P3TxxRdXZ599dr3s9bvf/W51zDHHVPfdd199bc4/5JBD6vPz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rTacMpWbfYZ/RTRtKGeWLNcaLDFunL2d1ihG+JllzWjz6g+3nTrnna+oPsz2rDin0/DKHTee9T+Od101+D5xrdIIbzGdhlnLqafubeqcifg/yUoppZRSSim1gBo3y8QIT2nvV9Q/QPXyt/5m9bbPblJ9eN9Nqnfv/qLq1ex75wYz39/9sz3Wq39karO/fX2188GbVR/+yvr1ea/e5XXTpnK+RnjL59Zx7rRi02r7v31hbb5fv8em08cXcdZp2PJ51Zs/O5WGKTO7854bVpttlc3xa6u3vGVqe9sNq+1XbFbtOrVv2tA/p3rj7htXH1m5RbXL/htXv1Mvl55tjpVSSimllFJqqWvcLB8jPKXdvvXq6r+949n1rzyvj7Z8dvXaHV5V7bgqH7dZ9f5d16teMWV+62M2f3a18U6vqT4aP0I1TyO88UdfWf02vxY9E+fG1c5x/Bpxlml4VrXRH7+y2j6lc4dP/ma1UR2+XvXO77Bvi+ojn35x9Sp+sXpwzgZbv6za9oA8k6yUUkoppZRSS1/jZlkZ4Rmt2qzaaTCT2hiOjti8+uiKN1W7zPwK83yVlzFvUe1ywKbVzrOMd4vqNMzh+Fqz/x2UUkoppZRSSi03jZvlaYQXVeX3eZVSSimllFJKrY3GjUZ4rbVxtdXbnltt+ok3NoQppZRSSimllJqrxo1GWCmllFJKKaXURGncaISVUkoppZRSSk2Uxo1GWCmllFJKKaXURGncaISVUkoppZRSSk2Uxo1GWCmllFJKKaXURGncaISVUkoppZRSSk2Uxo1GWCmllFJKKaXURGncTKwRFhERERERERkHGmERERERERHpFRphERERERER6RUaYREREREREekVGmERERERERHpFRphERERERER6RUaYREREREREekVGmERERERERHpFRphERERERER6RUaYREREREREekVGmERERERERHpFRphERERERER6RUaYREREREREekVS8oI3/rIE9Vjv/r3wZYsJfa65LbqrseeGmytyZX3PVatvPKuwdbS5YYHH1/QOnr3VJ6hcbIQ7Yr7furffz3YWrfYT4iIiIhIF0vKCL98v3Orb19+x2BrMvi3KYN3/xO/GmxJEzc+9Hj1H/7l9Or/nH/TYM+a+bb9CT+r/uv/O2uwtTS44M6Hq5NuvG+wNc1v/N/TFrSOvvPIS2qNk4Vo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tBAff8tj5poPJXFMXId751qHXz37V3RHTcdz9lxd/2U7YDubkzLfiDsbLIwQ3zfmOogls+wLyviAfW332Zb+8vrAvjBiTcRyXq4bhm+UdEU6yGfCtjr0J0PrB9fPacj5wn0QR5QTcfCr18T5X755Vp22Hb5/xazzWb7M/sgDyoH7zmluS19TPgHHRb0f1nba2kVX2ofV41HuR0RERESkZKKNMDBAvuHB5h+ZWmgYaMcAmmvGIH2+EEfMcHWBEWYgn38pORP5sLZpaiKbjK787koH97uQ6bx7ymyihaIr/YtFpGPU+lFCnpQ/SBY88MTTQ8MCwjluGGuTvq78bYt3lLQ30XU/IiIiIiKZiTfCi0k2wosJv5TLv/GJf+Wz2AybbRMREREREVmOTKwRxpgtpuD5e62uDrvq7sbwcen2R5+sr8vyU5baNh0zbvFvk0gDf5vClVoOEhEREREJnBEWERERERGRXqERFhERERERkV6hERYREREREZFeoREWERERERGRXqERFhERERERkV6hERYREREREZFeoREWERERERGRXqERFhERERERkV6hERYREREREZFeoREWERERERGRXqERFhERERERkV6hERYREREREZFeoREWERERERGRXqERFhERERERkV6hERYREREREZFeoREWERERERGRXqERFhERERERkV6hERYREREREZFeoREWERERERGRXqERFhERERERkV6hERYREREREZFeoREWERERERGRXqERFhERERERkV4xLyN89933VA8//MhgS0RERERERGRpgJfF0w5jqBF+8qmnagetGRYREREREZGlAh4WL4unHcZQIwyciIsmEqWUUkoppZRSatKFh20zwdBqhEVERERERESWGxphERERERER6RUaYREREREREekVGmERERERERHpFRphERERERER6RUaYREREREREekVGmERERERERHpFRphERERERER6RUaYREREREREekRVfX/ASKtR/OhSWGvAAAAAElFTkSuQmCC" alt="secondsubmit.PNG"></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>It can be seen that the K-Nearest Neighbors Classifier achieved an accuracy of 77.751% on the test data with K=25. This is a minor improvement on the predictions given by the Random Forest Classifier. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The working notebook can be found here <a href="https://colab.research.google.com/drive/1LcKbMLrBNCftdlnZcpJsqgynD6KZJyGh?usp=sharing">https://colab.research.google.com/drive/1LcKbMLrBNCftdlnZcpJsqgynD6KZJyGh?usp=sharing</a></p>

</div>
</div>
</div>
    </div>
  </div>
</body>
</html>