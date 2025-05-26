from rich.console import Console
from tqdm import tqdm

console = Console()

def print_start(title: str):
    console.print(f"[bold cyan]🐶 {title} started...[/bold cyan]")

def print_epoch_summary(epoch_index: int, average_loss: float):
    console.print(f"[bold blue]⚙️ Epoch {epoch_index} Summary[/bold blue]")
    console.print(f"[green]Average Training Loss:[/green] {average_loss:.4f}")

def print_validation_accuracy(accuracy: float, min_prob: float, max_prob: float):
    console.print(f"[bold green]✅ Validation Accuracy:[/bold green] {accuracy:.4f}")
    console.print(f"[dim]Probability range: {min_prob:.3f}–{max_prob:.3f}[/dim]")

def progress_bar(iterable, description: str):
    return tqdm(iterable, desc=description, ncols=80)

def print_success(message: str):
    console.print(f"[bold green]✅ {message}[/bold green]")

def print_warning(message: str):
    console.print(f"[bold yellow]⚠️ {message}[/bold yellow]")

def print_error(message: str):
    console.print(f"[bold red]❌ {message}[/bold red]")