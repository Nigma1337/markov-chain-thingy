#!/usr/bin/env python3
import glob
import typer
import markovify
import os
import csv
from tqdm import tqdm
app = typer.Typer()

@app.command()
def compile_discord(path: str, out: str = typer.Argument("model.json"), standalone: bool = typer.Argument(True)):
    if path.endswith("/"):
        path = path[:-1]
    corpus = ""
    for messages_csv in glob.iglob(path+"/**/messages.csv", recursive=True):
        with open(messages_csv, 'r') as f:
            reader = csv.reader(f)
            try:
                messages_column = next(reader).index("Contents")
            except StopIteration as e:
                continue
            for row in reader:
                corpus += row[messages_column]+"\n"
    if len(corpus) == 0:
        typer.echo(typer.style("Discord messages not found, please double check path, or. request it via steps here: https://support.discord.com/hc/en-us/articles/360004027692-Requesting-a-Copy-of-your-Data", fg=typer.colors.RED))
        return
    text_model = markovify.NewlineText(corpus)
    if standalone:
        text_model.compile(inplace=True)
        with open(out, "w+") as f:
            f.write(text_model.to_json())
        return
    return text_model

@app.command()
def compile_toxic(out: str = typer.Argument("model.json"), standalone: bool = typer.Argument(True)):
    corpus = ""
    if not os.path.exists("train_preprocessed.csv"):
        typer.echo(typer.style("", fg=typer.colors.RED))
    with open("train_preprocessed.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip headers
        for row in tqdm(reader):
            row[5] = 0
            #only include toxic comments
            if "1.0" in row[2:]:
                corpus += row[0]+"\n"
    if len(corpus) == 0:
        typer.echo(typer.style("CSV file with toxic comments not found", fg=typer.colors.RED))
        return
    text_model = markovify.NewlineText(corpus)
    if standalone:
        text_model.compile(inplace=True)
        with open(out, "w+") as f:
            f.write(text_model.to_json())
        return
    return text_model

@app.command()
def compile_all(path: str, out: str = typer.Argument("model.json")):
    toxic = compile_toxic(standalone=False)
    discord = compile_discord(path=path, standalone=False)
    final = markovify.combine([toxic, discord], [1, 3])
    final.compile(inplace=True)
    with open(out, "w+") as f:
        f.write(final.to_json())

@app.command()
def run(model_path: str = typer.Argument("model.json")):
    if not os.path.exists(model_path):
        typer.echo(typer.style(f"Model at {model_path} not found, try running one of the compile commands first", fg=typer.colors.RED))
    with open(model_path, "r") as f:
        model = markovify.Text.from_json(f.read())
    for i in range(10):
        typer.echo(model.make_sentence())

if __name__ == "__main__":
    app()