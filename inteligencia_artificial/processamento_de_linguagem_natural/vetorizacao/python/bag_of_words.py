import string

def conta_ocorrencias(palavra_escolhida: str, sentenca: str) -> int:
    contador = 0
    for palavra in sentenca.split():
        if palavra == palavra_escolhida:
            contador += 1

    return contador


def normaliza_sentenca(sentenca: str) -> str:
    return sentenca.translate(str.maketrans('', '', string.punctuation)).lower()


def constroi_bow(sentenca: str) -> dict[str, int]:
    sentenca = normaliza_sentenca(sentenca)
    palavras = sentenca.split()
    bow: dict[str, int] = {}
    for palavra in palavras:
        bow[palavra] = conta_ocorrencias(palavra, sentenca)
    
    return bow

if __name__ == "__main__":
    sentenca = input('Digite uma frase: ')
    print(constroi_bow(sentenca))
        