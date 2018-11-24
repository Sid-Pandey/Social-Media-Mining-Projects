#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 19:42:33 2018

@author: ronak
"""

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import unicodedata
import sys
from bs4 import BeautifulSoup

html_page = open("./amazon_search_latest.html",mode="r",encoding="utf-8")

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))


# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)

# initialize the stemmer
stemmer = LancasterStemmer()
# variable to hold the Json data read from the file
data = None

# read the json file and load the training data
with open('./values.json') as json_data:
    data = json.load(json_data)
    print(data)
    
# get a list of all categories to train for
categories = list(data.keys())
words = []
# a list of tuples with words in the sentence and category name
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        # remove any punctuation from the sentence
        each_sentence = remove_punctuation(each_sentence)
        print(each_sentence)
        # extract words from each sentence and append to the word list
        w = nltk.word_tokenize(each_sentence)
        print("tokenized words: ", w)
        words.extend(w)
        docs.append((w, each_category))
        
# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

print(words)
print(docs)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)

for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells
    # which catefory that bow belongs to.
    training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=150, batch_size=8, show_metric=True)
model.save('model.tflearn')

# a method that takes in a sentence and list of all words
# and returns the data in a form the can be fed to tensorflow


def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))

soup = BeautifulSoup(html_page, 'html.parser')
#print(soup.prettify())
recomm_list = 0

'''
BOTTOM UP TRAVERSAL : LOOKING AT A sg-product tag and then looking at parent.
'''
catg_map = {}
print("Looking for non category blocks first")
product_name = ""
product_category = ""

###NEW PIECE OF CODE THAT WAS ADDED. THIS IS ABLE TO TAKE THE TOP FEW CARDS ON THE PAGE.
for div in soup.find_all('div',class_="a-cardui"):
	print("\n\nStarting next div")
	for child in div.children:
		#print(child)
		if(child.name == "div"):		##get the cardui header element to know the title.
			class_list = child["class"]
			if(class_list[0] == "a-cardui-header"):   ##if it's a header class.
				header = child.find("h2")
				if(len(header) != 0):
					print("Extracted header : " + header.text)
					product_category = header.text
			if(class_list[0] == "a-cardui-body"):
				paras = child.find_all("p")
				img = child.find("img")
				span = child.find("span")
				if img != None:
					print("The text to be displayed : " , img["alt"])
					product_name = img["alt"]
				elif paras != None:
					for p in paras:
						print("The text to be displayed : " + p.text)
						if(p.text != ""):
							product_name = p.text
				else:
					print("The text to be displayed : " , span.text)
					if(span.text != ""):
						product_name = span.text
	print("Product name : " + product_name + " Product category : " + product_category)
	catg_map.update({product_name : product_category})

##NEW CODE THAT ADDS THE MISCELLANEOUS NON RECOMMENDATION SECTIONS WHICH OCCUR IN BETWEEN THE MAIN SECTIONS. THE VALUE FOR THE PRODUCT IS : Miscellaneous-Non Recommendation
for div in soup.find_all("div",class_="billboard"):
	product_category = div.a.img["alt"]
	print("Adding miscellaneous product "  + product_category)
	catg_map.update({"Miscellaneous Category" : product_category})
	
print("\n\nStarted looking for categories : \n")
for listitem in soup.find_all('li'):
    if(listitem.has_attr('data-sgproduct')):
        print("\n\nNow looking at the next item in list : \n")
    #    print("Product is : " + str(listitem))
        top_level_parent = ""
        product_category = ""
        catg_found = 0
        product_name = listitem.span.a.img["alt"] 
        print(product_name)
        for parent in listitem.parents:
            #print("Parent name is : " + parent.name)
            if(parent.name == "div"):
                all_spans = parent.find_all('span',class_='a-color-base')
                span_list = list(all_spans)    
                if(len(span_list) != 0):
                    catg_found = 1
                    product_category = span_list[0].get_text()
            if(catg_found == 1):
                break
            if(parent.name == 'html'):
                break
            top_level_parent = parent.name
        print("Product category is : " + product_category)
        catg_map.update({product_name : product_category})
		
p1_mayLike = 0
p2_rec = 0
p3_noRec = 0
p4_topPicks = 0
p5_blackFriday = 0
p6_bestSellers = 0
p7_relatedItems = 0
rec_category = []
chk = ''
values_list = []
for key, value in sorted(catg_map.items()):
    values_list.append(value)
    #print(key, value)
    if 'popular in brands you may like' in value.lower():
        p1_mayLike = p1_mayLike + 1
    elif 'recommend' in value.lower():
        p2_rec = p2_rec + 1
        if(value.lower().find("in") > -1):
            chk = value[value.lower().find("in") + 3:]
            rec_category.append(chk)
        else: rec_category.append(value)
    elif 'picks' in value.lower():
        p4_topPicks = p4_topPicks + 1
    elif 'friday' in value.lower():
        p5_blackFriday = p5_blackFriday+ 1
    elif 'best seller' in value.lower():
        p6_bestSellers = p6_bestSellers+ 1
    elif value.lower() == "related to items you've viewed":
        p7_relatedItems += 1
    else:
        p3_noRec = p3_noRec + 1

values_list1 = []
for i in values_list:
    j = i.replace('  ','').replace('\n','')
    values_list1.append(j) 

final_labeling = {}
final = []
all_labels = []
for tag in values_list1:
    #print(categories[np.argmax(model.predict([get_tf_record(tag)]))])
    all_labels.append(categories[np.argmax(model.predict([get_tf_record(tag)]))])
    final.append({ tag : categories[np.argmax(model.predict([get_tf_record(tag)]))] })
    final_labeling.update({ tag : categories[np.argmax(model.predict([get_tf_record(tag)]))] })
print(final)
print("\n")
output = {}
d = {x:all_labels.count(x) for x in all_labels}
print("Home Page Products Labels: " + str(d))
output.update({"product_labels": d})
print("\n")
val = {x:values_list1.count(x) for x in values_list1}
print("Product Category Quantities: " + str(val))
output.update({"category_quantities": val})
print("\n")
b = {x:rec_category.count(x) for x in rec_category}
print("Strong Recommendations category: " + str(b))
output.update({"reco_categories": b})
with open('output_21Nov_withTVandBagSearch.txt', 'w') as outfile:  
    json.dump(output, outfile)

