def conta_ocorrencias(palavra_escolhida: str, sentenca: str) -> int:
    contador = 0
    for palavra in sentenca:
        if palavra == palavra_escolhida:
            contador += 1

    return contador

if __name__ == "__main__":
    sentenca = input('Digite uma frase: ')
    palavras = sentenca.split()
    for palavra in palavras:
        print(f"{palavra}: {conta_ocorrencias(palavra, sentenca)}")