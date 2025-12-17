import string

def conta_ocorrencias(palavra_escolhida: str, sentenca: str) -> int:
    contador = 0
    for palavra in sentenca.split():
        if palavra == palavra_escolhida:
            contador += 1

    return contador

if __name__ == "__main__":
    sentenca = input('Digite uma frase: ')
    sentenca = sentenca.translate(str.maketrans('', '', string.punctuation))
    sentenca = sentenca.lower()
    palavras = sentenca.split()
    bow = {}
    for palavra in palavras:
        bow[palavra] = conta_ocorrencias(palavra, sentenca)

    print(bow)
        