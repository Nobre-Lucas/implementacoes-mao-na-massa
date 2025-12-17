from bag_of_words import constroi_bow, normaliza_sentenca

def term_frequency(palavra: str, documento: str) -> float:
    bow = constroi_bow(documento)
    return bow[palavra] / len(documento.split())


def inverse_document_frequency(palavra: str, corpus: list[str]) -> float:
    N = len(corpus)
    contador = 0
    for documento in corpus:
        for palavra_ in documento.lower().split():
            if palavra_ == palavra:
                contador += 1

    return N / contador


def calcula_tf_idf_palavra(palavra, documento, corpus):
    return term_frequency(palavra, documento) * inverse_document_frequency(palavra, corpus)


def tf_idf(corpus: list[str]) -> list[list[float]]:
    vetores = []
    for documento in corpus:
        vetor_documento = {}
        sentenca = normaliza_sentenca(documento)
        for palavra in sentenca.split():
            vetor_documento[palavra] = (calcula_tf_idf_palavra(palavra, documento, corpus))
        vetores.append(vetor_documento)

    return vetores


if __name__ == "__main__":
    qtd_docs = int(input('Informe o número de documentos: '))
    corpus = []
    for i in range(qtd_docs):
        documento = input('Digite um texto: ')
        corpus.append(documento)

    print(term_frequency('this', corpus[0]))
    print(inverse_document_frequency('this', corpus))
    print(tf_idf(corpus))
    
