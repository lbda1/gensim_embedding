#coding=utf-8
import xlrd

__author__ = 'liyang54'
import codecs
from gensim.models import Word2Vec

# 读文件
def read_file():
    data = xlrd.open_workbook(r"0718_training_dev_test.xlsx")
    table = data.sheets()[0]
    nrows = table.nrows
    print("-----------------------")
    print(nrows)
    user_query_list = []
    for i in range(nrows):
        every_line_word = []
        clo1 = table.row_values(i)[0].strip()
        clo2 = table.row_values(i)[1].strip()
        user_query_list.append(clo1)
        user_query_list.append(clo2)
        # every_line_word_one = clo1.split(" ")
        # every_line_word_two = clo2.split(" ")
        # user_query_list.append(every_line_word_one)
        # user_query_list.append(every_line_word_two)
    user_query_set_list = list(set(user_query_list))
    user_query_segment_list = []
    for element in user_query_set_list:
        every_question_word_list = element.split(" ")
        user_query_segment_list.append(every_question_word_list)
    return user_query_segment_list


def export_to_file(model, output_file):
    output = codecs.open(output_file, 'w' , 'utf-8')
    print('done loading Word2Vec')
    vocab = model.wv.vocab
    for mid in vocab:
        #print(model[mid])
        #print(mid)
        vector = list()
        for dimension in model[mid]:
            vector.append(str(dimension))
        #line = { "mid": mid, "vector": vector  }
        vector_str = " ".join(vector)
        line = mid + " "  + vector_str
        #line = json.dumps(line)
        output.write(line + "\n")
    output.close()

if __name__ == '__main__':
    user_query_list = read_file()
    # user_query_list是list的list，里面是分好词的句子
    model = Word2Vec(user_query_list, size=100, window=5, min_count=1, workers=4, iter=10)
    print(model.wv.most_similar("使用"))
    export_to_file(model,"word2vec_by_gensim_ly_train_dev_test_0823.txt")
