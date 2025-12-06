from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="es")

pred=analyzer.predict("Encontramos la calma en la serenidad de la rutina diaria")

print(pred)