import os
import sys
import subprocess
import datetime


def run_command(command):
    """Выполняет команду и выводит результат."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка выполнения: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def check_git_status():
    print("\n🔍 ТЕКУЩИЙ СТАТУС РЕПОЗИТОРИЯ:")
    print("-" * 40)
    run_command("git status --short")
    print("-" * 40)


def main():
    print("=" * 60)
    print("  🚀 GIT HELPER: ОТПРАВКА ИЗМЕНЕНИЙ В РЕПОЗИТОРИЙ")
    print("=" * 60)

    # Проверка наличия git
    if not os.path.exists(".git"):
        print("❌ Ошибка: Это не Git-репозиторий!")
        return

    # 1. Показываем статус
    check_git_status()

    # 2. Автоматическое добавление изменений (исключая docs_local благодаря .gitignore)
    print("\n📦 Добавление измененных файлов (игнорируя docs_local/)...")
    run_command("git add .")

    # 3. Повторный статус после добавления
    check_git_status()

    confirm = input("\n❓ Отправить эти изменения на GitHub? (y/n): ").strip().lower()
    if confirm != 'y':
        print("⛔ Отмена операции.")
        return

    # 4. Коммит
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    default_msg = f"update: автоматический коммит от {timestamp}"
    message = input(f"\n✏️ Введите сообщение коммита (Enter для '{default_msg}'): ").strip()
    if not message:
        message = default_msg

    print(f"\n💾 Создание коммита: '{message}'...")
    if not run_command(f'git commit -m "{message}"'):
        # Если ничего не изменилось
        if "nothing to commit" in str(subprocess.run("git status", shell=True, capture_output=True, text=True).stdout):
            print("ℹ️ Изменений для коммита нет.")
            return
        else:
            print("❌ Не удалось создать коммит.")
            return

    # 5. Push
    branch = subprocess.run("git rev-parse --abbrev-ref HEAD", shell=True, capture_output=True,
                            text=True).stdout.strip()
    print(f"\n🌐 Отправка в ветку '{branch}' на GitHub...")
    if run_command(f"git push origin {branch}"):
        print("\n✅ УСПЕШНО! Изменения отправлены на сервер.")
    else:
        print("\n❌ ОТПРАВКА НЕ УДАЛАСЬ. Проверьте подключение к интернету или права доступа.")


if __name__ == "__main__":
    main()
