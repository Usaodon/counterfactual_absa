import random


extra_patterns = {
    2:
        {
            'single': [
                '<originsent>, which is great!',
                '<originsent>, a very good experience'
            ],
            'multi': [
                '<originsent>, which is great for <term>!',
                '<originsent>, a very good experience of <term>'
            ],
        },
    1: {
            'single': [
                '<originsent>, which seems normal',
                '<originsent>, a rather ordinary experience'
            ],
            'multi': [
                '<originsent>, which seems normal for <term>',
                '<originsent>, a rather ordinary experience of <term>'
            ],
        },
    0: {
            'single': [
                '<originsent>, which is terrible!',
                '<originsent>, a very bad experience'
            ],
            'multi': [
                '<originsent>, which is terrible for <term>!',
                '<originsent>, a very bad experience of <term>'
            ],
        },
}


def add_patten(sent, label, pat_type, term):
    sentiments = [0, 1, 2]
    sentiments.remove(label)
    target_sent, target_polarity, pattern = [], [], []
    for polarity in sentiments:
        for temp_target_sent in extra_patterns[polarity][pat_type]:
            temp_pattern = temp_target_sent.replace('<originsent>', '').replace('<term>', term)
            temp_target_sent = temp_target_sent.replace('<originsent>', sent).replace('<term>', term)
            target_sent.append(temp_target_sent)
            target_polarity.append(polarity)
            pattern.append(temp_pattern)
    return target_sent, target_polarity, pattern


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def random_mask(sent, index, term, radio):
    index_range = list(range(0, len(sent)))

    for i in range(len(index)):
        if index[i] == 1.0:
            index_range.remove(i - 1)

    num = max(int(radio * len(sent)), 1)
    index = random.sample(index_range, num)
    index.sort()
    try:
        for i in index:
            sent[i] = '<extra_id_99>'
        sent[index[-1]] = '<extra_id_0>'
    except:
        pass
    return ' '.join(sent)
