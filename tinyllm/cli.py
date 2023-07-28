# cli.py
import click

@click.group()
def main():
    pass

@main.command()
def start():
    import tinyllm.app

if __name__ == "__main__":
    main()