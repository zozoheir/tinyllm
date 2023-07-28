# cli.py
import click

@click.group()
def main():
    pass

@main.command()
def start():
    click.echo('Running the start command.')
    # Call your function here


if __name__ == "__main__":
    main()
