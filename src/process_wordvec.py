import fasttext
from fasttext import load_model
from tqdm import tqdm


model = fasttext.train_unsupervised('../dataset/all.code', model='skipgram', dim=300, epoch=200)
model.save_model("../trained_models/code.bin")

# original BIN model loading
f = load_model('../trained_models/code.bin')
lines = []

# get all words from model
words = f.get_words()

with open('../word_vec/code.vec', 'w') as file_out:
    # the first line must contain number of total words and vector dimension
    file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")

    # line by line, you append vectors to VEC file
    for w in tqdm(words):
        v = f.get_word_vector(w)
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        try:
            file_out.write(w + vstr + '\n')
        except:
            pass

