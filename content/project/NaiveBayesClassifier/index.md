---
title: Naive Bayes Classifier For Sentiment Analysis.
tags:
- Machine Learning
date: "2016-04-27T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

---
<html>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><h1 style="text-align: center;">Naive Bayes Sentiment Classification</h1></p>
<p>This blog post will introduce the concept of the Naive Bayes Classifier and evaluate its predictive capabilities on the labelled imdb data for sentiment analysis. The dataset can be found on Kaggle at the following link, https://www.kaggle.com/marklvl/sentiment-labell. The methods for performing experimentation on the Naive Bayes Classifier are referenced from the Data Mining-5334 assignment found at https://docs.google.com/document/d/1bmCm9TXwqp5tX7lpg14NkaB3dBSg15cCC7ICxeB-vB4/edit. All code and experiment results are my contribution to the methodology given. Numpy is the primary package used for handling the data along with base python structures. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="kn">as</span> <span class="nn">np</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For the classifier that will be built in this blog, we only require the imdb_labelled.txt text file from Kaggle. As the classifier will predict for a given line of text the sentiment, we can simply just load the lines of the text file into an array as follows. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="n">data</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="s1">&#39;imdb_labelled.txt&#39;</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">f</span><span class="o">.</span><span class="n">readlines</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Next we will split our dataset into train/dev/test datasets. We will reserve 10% of the data for testing, this data will never be seen by any model and is solely used for final evaluation. The remaining 90% of the dataset are split into test and dev sets with 72% being used for training and 18% used for development/validation. The value 18% is used for development as it can split 90% evenly 5 ways which will be useful for 5-fold cross validation later. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="n">train_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[:</span><span class="mi">720</span><span class="p">]</span>

<span class="n">dev_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">720</span><span class="p">:</span><span class="mi">900</span><span class="p">]</span>

<span class="n">test_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">900</span><span class="p">:]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The classifier that will be used to predict the sentiment for this dataset will be the Naive Bayes Classifier. This classifier works by using Baye's theorem to calculate the probability of the sentiment being positive given a word. This is abastracted to allow for a full sentence or multiple words by making a naive assumption about the conditional independence of the words, P(Sentiment|word1, word2) = P(word1|Sentiment) * P(word2|sentiment) * P(Sentiment) / P(word1, word2). This assumption allows the probabilities of positive and negative sentiments given the sentence to be calculated and compared. The classifier then predicts the class that has the higher probability value. In the unlikely case of a tie, we arbitrarily choose negative sentiment as the prediction. In the implementation following we leave out the calculation of the denominator as this will be a constant between predicting the likelihood of positive sentiment vs negative sentiment and thus will not impact the classification results.</p> <p>Our classifier trains on the provided training data via the train function. This function takes the sentence as input, does some minor filtering (replace commas and periods with space and make all words lower case), and places the words from the sentence into a temporary set. As the words are in a set, we are only looking for a single occurrence of a word per line. The labels are included at the end of the input line of the training data and is parsed by the model and removed from the line as to not include the value in the dictionary. Finally the set is iterated over and each word is stored in the dictionary as a 3-dimensional array; this array stores the total count of the word in all documents, the count of the word in positive documents, and the count of the word in negative documents. If the word is already present in the dictionary the appropriate values of the array are incremented based on the sentiment label. The implementation as described above is provided below.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="k">class</span> <span class="nc">NBC</span><span class="p">():</span>
  <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">model_dict</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">pos_docs</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">neg_docs</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="bp">self</span><span class="o">.</span><span class="n">tot_docs</span> <span class="o">=</span> <span class="mi">0</span>

  <span class="k">def</span> <span class="nf">train</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">train_data</span><span class="p">):</span>
    <span class="k">for</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">train_data</span><span class="p">:</span>
      <span class="n">label</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">data</span><span class="p">)</span><span class="o">-</span><span class="mi">2</span><span class="p">]</span>
      <span class="n">positive</span> <span class="o">=</span> <span class="bp">True</span>
      <span class="n">data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[:</span><span class="nb">len</span><span class="p">(</span><span class="n">data</span><span class="p">)</span><span class="o">-</span><span class="mi">3</span><span class="p">]</span>
      <span class="n">words</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">split</span><span class="p">()</span>
      <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">words</span><span class="p">)):</span>
        <span class="n">words</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">words</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s2">&quot;.&quot;</span><span class="p">,</span><span class="s2">&quot;&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s2">&quot;,&quot;</span><span class="p">,</span><span class="s2">&quot;&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span>
      <span class="n">words</span> <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">words</span><span class="p">)</span>


      <span class="k">if</span> <span class="n">label</span> <span class="o">==</span> <span class="s1">&#39;1&#39;</span><span class="p">:</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">pos_docs</span> <span class="o">+=</span> <span class="mi">1</span>
      <span class="k">elif</span> <span class="n">label</span> <span class="o">==</span> <span class="s1">&#39;0&#39;</span><span class="p">:</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">neg_docs</span> <span class="o">+=</span> <span class="mi">1</span>
        <span class="n">positive</span> <span class="o">=</span> <span class="bp">False</span>
      <span class="bp">self</span><span class="o">.</span><span class="n">tot_docs</span> <span class="o">+=</span> <span class="mi">1</span>

      <span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">words</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">word</span> <span class="ow">not</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">model_dict</span><span class="p">:</span>
          <span class="k">if</span> <span class="n">positive</span><span class="p">:</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">model_dict</span><span class="p">[</span><span class="n">word</span><span class="p">]</span> <span class="o">=</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">]</span>
          <span class="k">else</span><span class="p">:</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">model_dict</span><span class="p">[</span><span class="n">word</span><span class="p">]</span> <span class="o">=</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">]</span>
        <span class="k">else</span><span class="p">:</span>
          <span class="bp">self</span><span class="o">.</span><span class="n">model_dict</span><span class="p">[</span><span class="n">word</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span> <span class="o">+=</span> <span class="mi">1</span>
          <span class="k">if</span> <span class="n">positive</span><span class="p">:</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">model_dict</span><span class="p">[</span><span class="n">word</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span> <span class="o">+=</span> <span class="mi">1</span>
          <span class="k">else</span><span class="p">:</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">model_dict</span><span class="p">[</span><span class="n">word</span><span class="p">][</span><span class="mi">2</span><span class="p">]</span> <span class="o">+=</span> <span class="mi">1</span>

  <span class="k">def</span> <span class="nf">word_probability</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">word</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">model</span><span class="o">.</span><span class="n">model_dict</span><span class="p">[</span><span class="n">word</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span><span class="o">/</span><span class="n">model</span><span class="o">.</span><span class="n">tot_docs</span>

  <span class="k">def</span> <span class="nf">word_given_positive</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">word</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">False</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">smoothing</span><span class="p">:</span>
      <span class="k">return</span> <span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">model_dict</span><span class="p">[</span><span class="n">word</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span><span class="o">/</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">pos_docs</span> <span class="o">+</span> <span class="mi">2</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
      <span class="k">return</span> <span class="n">model</span><span class="o">.</span><span class="n">model_dict</span><span class="p">[</span><span class="n">word</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span><span class="o">/</span><span class="n">model</span><span class="o">.</span><span class="n">pos_docs</span>

  <span class="k">def</span> <span class="nf">word_given_negative</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">word</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">False</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">smoothing</span><span class="p">:</span>
      <span class="k">return</span> <span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">model_dict</span><span class="p">[</span><span class="n">word</span><span class="p">][</span><span class="mi">2</span><span class="p">]</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span><span class="o">/</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">neg_docs</span> <span class="o">+</span> <span class="mi">2</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
      <span class="k">return</span> <span class="n">model</span><span class="o">.</span><span class="n">model_dict</span><span class="p">[</span><span class="n">word</span><span class="p">][</span><span class="mi">2</span><span class="p">]</span><span class="o">/</span><span class="n">model</span><span class="o">.</span><span class="n">neg_docs</span>

  <span class="k">def</span> <span class="nf">positive</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">word</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">False</span><span class="p">):</span>
    <span class="k">return</span> <span class="p">((</span><span class="bp">self</span><span class="o">.</span><span class="n">word_given_positive</span><span class="p">(</span><span class="n">word</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="n">smoothing</span><span class="p">))</span><span class="o">*</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">word_probability</span><span class="p">(</span><span class="n">word</span><span class="p">)))</span><span class="o">/</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">pos_docs</span><span class="o">/</span><span class="n">model</span><span class="o">.</span><span class="n">tot_docs</span><span class="p">)</span>

  <span class="k">def</span> <span class="nf">negative</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">word</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">False</span><span class="p">):</span>
    <span class="k">return</span> <span class="p">((</span><span class="bp">self</span><span class="o">.</span><span class="n">word_given_negative</span><span class="p">(</span><span class="n">word</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="n">smoothing</span><span class="p">))</span><span class="o">*</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">word_probability</span><span class="p">(</span><span class="n">word</span><span class="p">)))</span><span class="o">/</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">neg_docs</span><span class="o">/</span><span class="n">model</span><span class="o">.</span><span class="n">tot_docs</span><span class="p">)</span>

  <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">test_data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">False</span><span class="p">):</span>
    <span class="n">words</span> <span class="o">=</span> <span class="n">test_data</span><span class="o">.</span><span class="n">split</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">words</span><span class="p">)):</span>
        <span class="n">words</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">words</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s2">&quot;.&quot;</span><span class="p">,</span><span class="s2">&quot;&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s2">&quot;,&quot;</span><span class="p">,</span><span class="s2">&quot;&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span>
    <span class="n">words</span> <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">words</span><span class="p">)</span>
    <span class="n">pos</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">pos_docs</span><span class="o">/</span><span class="n">model</span><span class="o">.</span><span class="n">tot_docs</span>
    <span class="n">neg</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">neg_docs</span><span class="o">/</span><span class="n">model</span><span class="o">.</span><span class="n">tot_docs</span>
    <span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">words</span><span class="p">:</span>
      <span class="k">if</span> <span class="n">word</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">model_dict</span><span class="p">:</span>
        <span class="n">pos</span> <span class="o">=</span> <span class="n">pos</span><span class="o">*</span><span class="bp">self</span><span class="o">.</span><span class="n">word_given_positive</span><span class="p">(</span><span class="n">word</span><span class="p">,</span> <span class="n">smoothing</span><span class="p">)</span>
        <span class="n">neg</span> <span class="o">=</span> <span class="n">neg</span><span class="o">*</span><span class="bp">self</span><span class="o">.</span><span class="n">word_given_negative</span><span class="p">(</span><span class="n">word</span><span class="p">,</span> <span class="n">smoothing</span><span class="p">)</span>
      <span class="k">else</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">smoothing</span><span class="p">:</span>
          <span class="n">pos</span> <span class="o">=</span> <span class="n">pos</span><span class="o">*</span><span class="p">(</span><span class="mi">1</span><span class="o">/</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">pos_docs</span> <span class="o">+</span> <span class="mi">2</span><span class="p">))</span>
          <span class="n">neg</span> <span class="o">=</span> <span class="n">neg</span><span class="o">*</span><span class="p">(</span><span class="mi">1</span><span class="o">/</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">neg_docs</span> <span class="o">+</span> <span class="mi">2</span><span class="p">))</span>
        <span class="k">else</span><span class="p">:</span>
          <span class="n">pos</span> <span class="o">=</span> <span class="mi">0</span>
          <span class="n">neg</span> <span class="o">=</span> <span class="mi">0</span>

    <span class="k">return</span> <span class="n">pos</span><span class="p">,</span> <span class="n">neg</span>

<span class="c1">#Return True for positive sentiment, False for negative sentiment</span>
  <span class="k">def</span> <span class="nf">predict</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">test_data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">False</span><span class="p">):</span>
    <span class="n">pos</span><span class="p">,</span> <span class="n">neg</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">forward</span><span class="p">(</span><span class="n">test_data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="n">smoothing</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">pos</span> <span class="o">&gt;</span> <span class="n">neg</span><span class="p">:</span>
      <span class="k">return</span> <span class="bp">True</span>
    <span class="k">else</span><span class="p">:</span>
      <span class="k">return</span> <span class="bp">False</span>

  <span class="k">def</span> <span class="nf">test</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">test_data</span><span class="p">,</span> <span class="n">labels</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">False</span><span class="p">):</span>
    <span class="n">correct</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">total</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">test_data</span><span class="p">)):</span>
      <span class="k">if</span> <span class="n">labels</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">==</span> <span class="s1">&#39;1&#39;</span><span class="p">:</span>
        <span class="n">label</span> <span class="o">=</span> <span class="bp">True</span>
      <span class="k">else</span><span class="p">:</span>
        <span class="n">label</span> <span class="o">=</span> <span class="bp">False</span>
      <span class="n">prediction</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test_data</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">smoothing</span><span class="o">=</span><span class="n">smoothing</span><span class="p">)</span>
      <span class="k">if</span> <span class="n">label</span> <span class="o">==</span> <span class="n">prediction</span><span class="p">:</span>
        <span class="n">correct</span> <span class="o">+=</span> <span class="mi">1</span>
      <span class="n">total</span> <span class="o">+=</span> <span class="mi">1</span>

    <span class="k">return</span> <span class="n">correct</span><span class="o">/</span><span class="n">total</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now we can initialize our model and train it on the training data from our previous split. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">NBC</span><span class="p">()</span>
<span class="n">model</span><span class="o">.</span><span class="n">train</span><span class="p">(</span><span class="n">train_data</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now that the model is trained, we can calculate some probabilities based on the given data and see what our model has learned. First we will look at the probability of the occurrence of the word 'the'. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="c1">#P(&#39;the&#39;)</span>
<span class="n">prob</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">word_probability</span><span class="p">(</span><span class="s1">&#39;the&#39;</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">prob</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>0.49166666666666664
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We get a result of .4917 or a 49.17% chance of the word 'the' occuring in a sentence. Next we will look at the probability of the word 'the' appearing given that the document (sentence) is of positive sentiment as well as the probability of 'the' occuring given a negative document.  </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="c1">#P(&#39;the&#39;|Positive)</span>
<span class="n">prob</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">word_given_positive</span><span class="p">(</span><span class="s1">&#39;the&#39;</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">prob</span><span class="p">)</span>

<span class="c1">#P(&#39;the&#39;|Negative)</span>
<span class="n">prob</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">word_given_negative</span><span class="p">(</span><span class="s1">&#39;the&#39;</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">prob</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>0.47112462006079026
0.5089514066496164
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We get a result of .4711 or a 47.11% chance of the word 'the' appearing given a document's sentiment is positive and .5173 or a 51.73% chance of 'the' appearing given a document's sentiment is negative. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As the model only learns on the 72% of data that is reserved for testing, it may not have seen some words that will be presented to it in the development of test datasets. This will cause the probability of the sentiment given this word to be zero and the classifier will not be able to account for any other words in the sentence. One approach to fixing this issue is to perform cross validation to find which training dataset results in the best predictions. We will perform 5-fold cross validation on our model by splitting the 90% of the dataset reserved for train/dev evenly into 5 folds. The model will take 1 fold (18% of data) as validation and the remaining 4 folds as the training data. The model will then be validated by calculating the accuracy of predictions on the validation set for all 5 models to be trained. The implementation can be seen below.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="k">def</span> <span class="nf">validate</span><span class="p">(</span><span class="n">train_data</span><span class="p">,</span> <span class="n">dev_data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">False</span><span class="p">):</span>
  <span class="n">model</span> <span class="o">=</span> <span class="n">NBC</span><span class="p">()</span>
  <span class="n">model</span><span class="o">.</span><span class="n">train</span><span class="p">(</span><span class="n">train_data</span><span class="p">)</span>
  <span class="n">labels</span> <span class="o">=</span> <span class="p">[]</span>
  <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">dev_data</span><span class="p">:</span>
    <span class="n">labels</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">x</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">x</span><span class="p">)</span><span class="o">-</span><span class="mi">2</span><span class="p">])</span>
    
  <span class="k">return</span> <span class="n">model</span><span class="o">.</span><span class="n">test</span><span class="p">(</span><span class="n">dev_data</span><span class="p">,</span> <span class="n">labels</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="n">smoothing</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="k">def</span> <span class="nf">FiveFoldCrossValidation</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">False</span><span class="p">):</span>
  <span class="n">accuracy</span> <span class="o">=</span> <span class="p">[]</span>
  <span class="n">train_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">180</span><span class="p">:</span><span class="mi">900</span><span class="p">]</span>
  <span class="n">dev_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[:</span><span class="mi">180</span><span class="p">]</span>

  <span class="n">accuracy</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">validate</span><span class="p">(</span><span class="n">train_data</span><span class="p">,</span> <span class="n">dev_data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="n">smoothing</span><span class="p">))</span>

  <span class="n">train_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[:</span><span class="mi">180</span><span class="p">]</span> <span class="o">+</span> <span class="n">data</span><span class="p">[</span><span class="mi">360</span><span class="p">:</span><span class="mi">900</span><span class="p">]</span>
  <span class="n">dev_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">180</span><span class="p">:</span><span class="mi">360</span><span class="p">]</span>

  <span class="n">accuracy</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">validate</span><span class="p">(</span><span class="n">train_data</span><span class="p">,</span> <span class="n">dev_data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="n">smoothing</span><span class="p">))</span>

  <span class="n">train_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[:</span><span class="mi">360</span><span class="p">]</span> <span class="o">+</span> <span class="n">data</span><span class="p">[</span><span class="mi">540</span><span class="p">:</span><span class="mi">900</span><span class="p">]</span>
  <span class="n">dev_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">360</span><span class="p">:</span><span class="mi">540</span><span class="p">]</span>

  <span class="n">accuracy</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">validate</span><span class="p">(</span><span class="n">train_data</span><span class="p">,</span> <span class="n">dev_data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="n">smoothing</span><span class="p">))</span>

  <span class="n">train_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[:</span><span class="mi">540</span><span class="p">]</span> <span class="o">+</span> <span class="n">data</span><span class="p">[</span><span class="mi">720</span><span class="p">:</span><span class="mi">900</span><span class="p">]</span>
  <span class="n">dev_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">540</span><span class="p">:</span><span class="mi">720</span><span class="p">]</span>

  <span class="n">accuracy</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">validate</span><span class="p">(</span><span class="n">train_data</span><span class="p">,</span> <span class="n">dev_data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="n">smoothing</span><span class="p">))</span>

  <span class="n">train_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[:</span><span class="mi">720</span><span class="p">]</span>
  <span class="n">dev_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">720</span><span class="p">:</span><span class="mi">900</span><span class="p">]</span>

  <span class="n">accuracy</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">validate</span><span class="p">(</span><span class="n">train_data</span><span class="p">,</span> <span class="n">dev_data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="n">smoothing</span><span class="p">))</span>

  <span class="k">return</span> <span class="n">accuracy</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="n">accuracy</span> <span class="o">=</span> <span class="n">FiveFoldCrossValidation</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">accuracy</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[0.6666666666666666, 0.65, 0.6277777777777778, 0.5777777777777777, 0.45555555555555555]
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>After performing 5-fold cross validation we get the resulting accuracies of our 5 models as [0.6666666666666666, 0.65, 0.6277777777777778, 0.5777777777777777, 0.45555555555555555]. We can see that the first split where the first 18% of the dataset is reserved for validation performs the best. Our accuracy values are still fairly low and when inspecting the model to determine why this is we see that it is due to the problem of unseen words resulting in zero probabilities. As this issue is still not solved after optimizing our training dataset, we will need to look into another method of solving this. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In order to fix the zero probability problem for unseen words we implement the technique known as smoothing. Smoothing allows us to artificially add 2 new documents for positive sentiment where one document contains all words in the dictionary and the other contains no words in the dictionary (as this counts for all possibilities for every attribute since a word can either be included or discluded). The same is done for the negative sentiment as well. As the documents are not real, they are not actually added to the dataset. Instead the documents are accounted for in the calculation of the probability for a word given the sentiment where the numerator (number of times a word appeared in documents of the given sentiment) is increased by one and the denominator (number of documents of the given sentiment) is increased by 2. After implementing smoothing, we perform 5-fold cross validation again to get the optimal hyperparameters for our model. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="n">smoothing_accuracy</span> <span class="o">=</span> <span class="n">FiveFoldCrossValidation</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">True</span><span class="p">)</span>
<span class="k">print</span><span class="p">(</span><span class="n">smoothing_accuracy</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[0.8666666666666667, 0.8611111111111112, 0.8666666666666667, 0.9, 0.7277777777777777]
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>After performing cross validation we get the resulting accuracies of the 5 models to be [0.8666666666666667, 0.8611111111111112, 0.8666666666666667, 0.9, 0.7277777777777777]. We can see that all of our models perform much better now that we are using the smoothing technique. Our best model gave a 90% accuracy on the validation set as the 4th fold. We use this model as our best and find the top ten words that predict positive and negative sentiments as well as their probabilities (P(Sentiment|word)). </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="n">train_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[:</span><span class="mi">540</span><span class="p">]</span> <span class="o">+</span> <span class="n">data</span><span class="p">[</span><span class="mi">720</span><span class="p">:</span><span class="mi">900</span><span class="p">]</span>
<span class="n">dev_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="mi">540</span><span class="p">:</span><span class="mi">720</span><span class="p">]</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">NBC</span><span class="p">()</span>
<span class="n">model</span><span class="o">.</span><span class="n">train</span><span class="p">(</span><span class="n">train_data</span><span class="p">)</span>

<span class="n">pos</span> <span class="o">=</span> <span class="p">[]</span>
<span class="n">neg</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">model</span><span class="o">.</span><span class="n">model_dict</span><span class="p">:</span>
  <span class="n">pos</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="n">model</span><span class="o">.</span><span class="n">positive</span><span class="p">(</span><span class="n">word</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">True</span><span class="p">),</span> <span class="n">word</span><span class="p">))</span>
  <span class="n">neg</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="n">model</span><span class="o">.</span><span class="n">negative</span><span class="p">(</span><span class="n">word</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">True</span><span class="p">),</span> <span class="n">word</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="k">def</span> <span class="nf">sort_help</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
  <span class="k">return</span> <span class="n">x</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>

<span class="n">pos</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">reverse</span><span class="o">=</span><span class="bp">True</span><span class="p">,</span> <span class="n">key</span><span class="o">=</span><span class="n">sort_help</span><span class="p">)</span>
<span class="n">neg</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">reverse</span><span class="o">=</span><span class="bp">True</span><span class="p">,</span> <span class="n">key</span><span class="o">=</span><span class="n">sort_help</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="k">print</span><span class="p">(</span><span class="n">pos</span><span class="p">[:</span><span class="mi">10</span><span class="p">])</span>
<span class="k">print</span><span class="p">(</span><span class="n">neg</span><span class="p">[:</span><span class="mi">10</span><span class="p">])</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[(0.5334168650545044, &#39;the&#39;), (0.2501942112517228, &#39;a&#39;), (0.24207492795389043, &#39;and&#39;), (0.1912709351376185, &#39;of&#39;), (0.18322683038884016, &#39;is&#39;), (0.15394896211836445, &#39;this&#39;), (0.115774965543165, &#39;i&#39;), (0.10162469197677818, &#39;it&#39;), (0.09197677818151442, &#39;to&#39;), (0.08896128304723717, &#39;in&#39;)]
[(0.506206896551724, &#39;the&#39;), (0.17875862068965517, &#39;a&#39;), (0.1724491600353669, &#39;and&#39;), (0.16499381078691425, &#39;of&#39;), (0.14500442086648982, &#39;is&#39;), (0.12767462422634834, &#39;this&#39;), (0.09687002652519892, &#39;i&#39;), (0.09276038903625108, &#39;it&#39;), (0.06930503978779841, &#39;in&#39;), (0.06878160919540229, &#39;to&#39;)]
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see the top ten words for positive sentiment and the probability P(Positive|word) are [(0.5334168650545044, 'the'), (0.2501942112517228, 'a'), (0.24207492795389043, 'and'), (0.1912709351376185, 'of'), (0.18322683038884016, 'is'), (0.15394896211836445, 'this'), (0.115774965543165, 'i'), (0.10162469197677818, 'it'), (0.09197677818151442, 'to'), (0.08896128304723717, 'in')]
. The top ten words for negative sentiment and the probability P(Negative|word) are [(0.506206896551724, 'the'), (0.17875862068965517, 'a'), (0.1724491600353669, 'and'), (0.16499381078691425, 'of'), (0.14500442086648982, 'is'), (0.12767462422634834, 'this'), (0.09687002652519892, 'i'), (0.09276038903625108, 'it'), (0.06930503978779841, 'in'), (0.06878160919540229, 'to')].
 </p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>It can be seen from the results that the same set of words are the top predictors for both positive and negative sentiment. This is because these are very common words in the English language. These words are included in a set of words called "Stop Words" and these words are typically filtered out of the dataset. Although they are typically filtered from models, our model still performs well with them included as they influence the positive and negative sentiment predictions near equally. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Finally we evaluate our best model with the hyperparameters found from the 5-fold cross validation on the last 10% of the dataset that was withheld for testing. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-python"><pre><span></span><span class="n">test_labels</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">test_data</span><span class="p">:</span>
  <span class="n">test_labels</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">x</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">x</span><span class="p">)</span><span class="o">-</span><span class="mi">2</span><span class="p">])</span>
<span class="n">model</span><span class="o">.</span><span class="n">test</span><span class="p">(</span><span class="n">test_data</span><span class="p">,</span> <span class="n">test_labels</span><span class="p">,</span> <span class="n">smoothing</span><span class="o">=</span><span class="bp">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[15]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0.71</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The final model accuracy on the test data is .71 or 71%. The accuracy is pretty decent taking into consideration the naive assumption being made and that the data being predicted on has never been seen by any of the models trained in the development process. </p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><h2>References</h2></p>
<p>Assignment/Methods. https://docs.google.com/document/d/1bmCm9TXwqp5tX7lpg14NkaB3dBSg15cCC7ICxeB-vB4/edit </p>
<p>Kaggle IMDB labelled dataset. https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set </p>
<p>Github. https://github.com/jjames71396/NaiveBayesClassifier </p>
<p>Working Notebook. https://colab.research.google.com/drive/1XuhBSHC48XSjhW6L2XwD3lI_5Z9lT5Pg?usp=sharing  </p>
</div>
</div>
</div>
    </div>
  </div>
</body>

</html>