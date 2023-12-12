# cli.py
import click

@click.group()
def main():
    pass

@main.command()
def agent():
    click.echo("Available soon!")

if __name__ == "__main__":
    main()