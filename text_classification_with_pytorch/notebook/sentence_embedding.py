
def sentence_vec(text, embedding_dict, stopwords, tokenizer):
    """
    Given a sentence and other parameters this function returns embeddings for whole sentence.
    1. Takes the sent and performed pre-processing
    2. For each word from the sent pull_out vectors associated with the word and store inthe list.
    3. List would be like : [x1, x2, x3, x4....... x300---->> for word1]
                            [x1, x2, x3, x4....... x300---->> for word2]
                            [x1, x2, x3, x4....... x300---->> for word3]
                            
                            
    : param text : any input sentence
    : param embedding_dict : dict {word:vector}
    : param stopwords : list of stopswords
    : param tokenizer : a tokenization func
    """
    
    # converting the text to the lower case
    words = str(text).lower()
    
    # tokenization of the sentence
    words = tokenizer(words)
    
    # removing the stopwords from the words 
    words = [word for word in words if word not in stopwords]
    
    # keeping only alpha-numeric tokens
    words = [word for word in words if word.isalpha()]
      
    M = []
    
    for word in words:
        # if word as key in embedding_dict then store its value in list.
        if word in embedding_dict:
            M.append(embedding_dict[word])
            
    if len(M) ==0:
        return np.zeros(300)
    
    M = np.array(M)
    
    print("Array storing the embeddings", M)
    
    # calculate sum along row, i.e for each sentences
    v = M.sum(axis=0)
    
    # Normalizing the vector
    return v/np.sqrt((v**2).sum())