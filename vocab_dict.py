
def get_dict():
    vocab_file = "/data/users/shihuima/sentence/character.vocab"
    dic = []
    with open(vocab_file) as f:
        for _, line in enumerate(f):
            if line[1:-2].split("_")[0] not in dic:
                dic.append(line[1:-2].split("_")[0])
    return dic,len(dic)


if __name__ == "__main__":
    get_dict()
