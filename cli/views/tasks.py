"""
cli/views/tasks.py
태스크 목록, 액션 다이얼로그, 신규 태스크 시작 (API-First 구조)
LLM Fine-tuning 전용으로 STT-Trainer 패턴 계승
"""
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.text import Text
from rich import box

from cli.client.api_client import api_client

console = Console()


def _get_status_style(status: str) -> str:
    return {
        "COMPLETED": "bold green",
        "RUNNING": "bold yellow",
        "FAILED": "bold red",
        "STOPPED": "bold magenta",
        "WAITING": "dim"
    }.get(status, "dim")


def show_task_list():
    """태스크 목록 표시 및 액션 메뉴"""
    while True:
        console.clear()
        console.print(Panel("[bold cyan]📋 태스크 관리 (원격 서버 연결됨)[/bold cyan]", expand=False))

        res = api_client.get("/api/v1/tasks/list")
        if "error" in res:
            console.print(f"[red]서버 통신 오류: {res['error']}[/red]")
            Prompt.ask("\n[dim]엔터를 눌러 돌아가기[/dim]")
            return

        tasks = res.get("tasks", [])
        if not tasks:
            console.print("[yellow]등록된 태스크가 없습니다.[/yellow]")
            Prompt.ask("\n[dim]엔터를 눌러 돌아가기[/dim]")
            return

        table = Table(box=box.ROUNDED, show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("태스크 명", style="cyan", min_width=20)
        table.add_column("ID (short)", style="dim", width=10)
        table.add_column("모델", style="magenta", width=25)
        table.add_column("상태", width=12)
        table.add_column("진행률", justify="right", width=8)
        table.add_column("생성일시", style="dim", width=20)

        for i, t in enumerate(tasks, 1):
            status_text = Text(t.get('status', 'N/A'), style=_get_status_style(t.get('status', '')))
            progress = f"{t.get('progress', 0):.1f}%"
            table.add_row(
                str(i),
                t.get('tsk_nm', 'N/A'),
                t.get('task_id', '')[:8],
                t.get('model_id', 'N/A'),
                status_text,
                progress,
                t.get('create_dt', 'N/A')
            )

        console.print(table)
        console.print("\n[dim]태스크 번호를 입력하면 액션을 선택합니다. [bold]0[/bold] = 메인으로[/dim]")

        choices = [str(i) for i in range(len(tasks) + 1)]
        pick = Prompt.ask("번호 선택", choices=choices, default="0")
        if pick == "0":
            return

        task = tasks[int(pick) - 1]
        _show_task_action_menu(task)


def _show_task_action_menu(task: dict):
    """개별 태스크에 대한 액션 메뉴"""
    from cli.views.monitor import watch_logs

    task_id = task.get('task_id', '')
    status = task.get('status', 'FAILED')
    name = task.get('tsk_nm', 'N/A')

    while True:
        console.clear()
        status_style = _get_status_style(status)

        console.print(Panel(
            f"[bold]{name}[/bold]\n"
            f"ID: [dim]{task_id}[/dim]\n"
            f"모델: [cyan]{task.get('model_id', 'N/A')}[/cyan]\n"
            f"상태: [{status_style}]{status}[/{status_style}]  |  진행률: {task.get('progress', 0):.1f}%",
            title="🎯 태스크 액션 (API)", expand=False, border_style="cyan"
        ))

        options = []

        if status == "RUNNING":
            options.append(("s", "⛔  강제 종료 (서버 킬)", "stop"))

        if status in ("FAILED", "STOPPED"):
            options.append(("r", "🔄  처음부터 재수행", "retry"))

        options.append(("v", "🔍  서버 원격 리포트/로그 확인", "view_report"))
        options.append(("l", "📡  실시간 로그/차트 모니터링", "logs"))
        options.append(("0", "◀  돌아가기", "back"))

        for key, label, _ in options:
            console.print(f"  [bold cyan]{key}[/bold cyan]  {label}")

        action_map = {o[0]: o[2] for o in options}
        pick = Prompt.ask("\n선택", choices=list(action_map.keys()), default="0")
        action = action_map[pick]

        if action == "back":
            return
        elif action == "logs":
            watch_logs(task_id)
        elif action == "stop":
            if Confirm.ask("[bold red]서버에서 프로세스를 강제 종료할까요?[/bold red]", default=False):
                res = api_client.post("/api/v1/tasks/stop", {"task_id": task_id})
                console.print(f"[yellow]종료 요청 결과: {res}[/yellow]")
                Prompt.ask("\n[dim]엔터[/dim]")
                return
        elif action == "retry":
            _retry_task(task)
            return
        elif action == "view_report":
            _show_report(task_id)


def _show_report(task_id: str):
    """원격 리포트 조회"""
    console.clear()
    console.print(Panel("[bold]📜 원격 리포트 요청 중...[/bold]", expand=False))

    res = api_client.get(f"/api/v1/tasks/report?task_id={task_id}")
    if "error" in res:
        console.print(f"[red]오류: {res['error']}[/red]")
        Prompt.ask("\n[dim]엔터[/dim]")
        return

    task = res.get("task_info", {})
    logs = res.get("logs", [])

    table = Table(title=f"원격 태스크 요약: {task.get('tsk_nm', 'N/A')}", box=box.ROUNDED)
    table.add_column("항목", style="cyan", width=20)
    table.add_column("값", style="white")
    table.add_row("상태", task.get('status', 'N/A'))
    table.add_row("진행률", f"{task.get('progress', 0):.1f}%")
    table.add_row("모델", task.get('model_id', 'N/A'))
    table.add_row("데이터 경로", task.get('data_path', 'N/A'))
    table.add_row("생성일시", task.get('create_dt', 'N/A'))
    table.add_row("시작일시", task.get('start_time', 'N/A'))
    table.add_row("종료일시", task.get('end_time', 'N/A'))
    console.print(table)

    console.print(f"\n[dim]최근 로그 (최대 20개):[/dim]")
    for log in logs[-20:]:
        level = log.get('level', 'INFO')
        style = {"ERROR": "red", "WARNING": "yellow", "SUCCESS": "green"}.get(level, "white")
        console.print(f"  [{style}][{level}][/{style}] {log.get('message', '')}")

    Prompt.ask("\n[dim]엔터를 눌러 돌아가기[/dim]")


def _retry_task(task: dict):
    """태스크 재시작"""
    console.clear()
    console.print(Panel("[bold yellow]🔄 태스크 재시작[/bold yellow]", expand=False))

    data_path = task.get('data_path', '')
    model_id = task.get('model_id', '')

    body = {
        "name": task.get('tsk_nm', 'Retry_Task'),
        "data_path": data_path,
        "model_id": model_id
    }

    res = api_client.post("/api/v1/tasks/create", body)
    if "error" in res:
        console.print(f"[red]오류: {res['error']}[/red]")
    else:
        new_task_id = res.get("task_id")
        console.print(f"[bold green]✅ 새 태스크 생성! Task ID: {new_task_id}[/bold green]")

        from cli.views.monitor import watch_logs
        if Confirm.ask("\n실시간 로그 모니터링 시작?", default=True):
            watch_logs(new_task_id)

    Prompt.ask("\n[dim]엔터[/dim]")


def start_new_task():
    """신규 학습 파이프라인 시작"""
    from cli.views.monitor import watch_logs

    console.clear()
    console.print(Panel("[bold green]🚀 신규 LLM 학습 파이프라인 (원격 기동)[/bold green]", expand=False))

    task_name = Prompt.ask("[cyan]1.[/cyan] 태스크 명", default="LLM_FineTune_Task")

    console.print("\n[dim]사용 가능한 모델:[/dim]")
    console.print("  [cyan]1[/cyan]  Qwen/Qwen2.5-0.5B-Instruct (추천 - 경량)")
    console.print("  [cyan]2[/cyan]  Qwen/Qwen2.5-1.5B-Instruct")
    console.print("  [cyan]3[/cyan]  직접 입력")

    model_choice = Prompt.ask("[cyan]2.[/cyan] 모델 선택", choices=["1", "2", "3"], default="1")
    if model_choice == "1":
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    elif model_choice == "2":
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    else:
        model_id = Prompt.ask("모델 ID (HuggingFace)", default="Qwen/Qwen2.5-0.5B-Instruct")

    data_path = Prompt.ask(
        "[cyan]3.[/cyan] 학습 데이터 경로 (ChatML JSONL)",
        default="dataset/train.jsonl"
    )

    body = {
        "name": task_name,
        "data_path": data_path,
        "model_id": model_id
    }

    console.print("\n[bold yellow]원격 서버로 파이프라인 기동 요청 중...[/bold yellow]")
    res = api_client.post("/api/v1/tasks/create", body)
    if "error" in res:
        console.print(f"[red]오류: {res['error']}[/red]")
    else:
        task_id = res.get("task_id")
        console.print(f"[bold green]✅ 기동 성공! Task ID: {task_id}[/bold green]")
        if Confirm.ask("\n실시간 로그 모니터링 시작?", default=True):
            watch_logs(task_id)

    Prompt.ask("\n[dim]엔터[/dim]")
