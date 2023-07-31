# cli.py
import click

@click.group()
def main():
    pass

@main.command()
def agent():
    import tinyllm.copilot_agent

if __name__ == "__main__":
    main()