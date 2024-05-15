from flask import Flask, request, render_template, redirect, url_for, session, jsonify, send_file
from flask_socketio import SocketIO, emit
import os
import zipfile
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_config import OpenAIConfig
import markdown
from markupsafe import Markup

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 使用一个随机的密钥
socketio = SocketIO(app)
UPLOAD_FOLDER = 'uploads'
SUMMARY_FOLDER = 'summaries'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)
analyzing = False
total_files = 0


class ProjectAnalyzer:
    def __init__(self, openai_config, project_path=None, project_zip=None, output_language='en'):
        self.openai_config = openai_config
        self.project_path = project_path
        self.project_zip = project_zip
        self.output_language = output_language

    def extract_zip(self):
        if self.project_zip:
            with zipfile.ZipFile(self.project_zip, 'r') as zip_ref:
                self.project_path = os.path.join(UPLOAD_FOLDER, os.path.basename(self.project_zip).replace('.zip', ''))
                zip_ref.extractall(self.project_path)

    def analyze_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()

        prompt = f"Analyze the following Python code and provide a summary in {self.output_language}:\n\n{code}"
        response = self.openai_config.chat(prompt)

        return response

    def analyze_project(self, max_concurrency):
        if self.project_zip:
            self.extract_zip()

        summaries = {}
        global total_files
        files = list(glob.glob(f"{self.project_path}/**/*.py", recursive=True))
        total_files = len(files)

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(self.analyze_file, file): file for file in files}
            for future in as_completed(futures):
                if not analyzing:
                    break
                file = futures[future]
                try:
                    summaries[file] = future.result()
                    index = len(summaries)
                    socketio.emit('progress', {'file': file, 'index': index, 'total': total_files})
                except Exception as e:
                    socketio.emit('error', {'message': str(e)})
                    return None, None

        if analyzing:
            project_summary = self.openai_config.chat(
                f"Analyze the following Python project and provide a summary based on these modules in {self.output_language}:\n\n{summaries}"
            )
        else:
            project_summary = "Analysis stopped."

        return project_summary, summaries

    def save_summary_as_markdown(self, project_summary, module_summaries):
        markdown_content = f"# Project Summary\n\n{project_summary}\n\n## Module Summaries\n"
        for module, summary in module_summaries.items():
            markdown_content += f"\n### {module}\n{summary}\n"

        summary_file = os.path.join(SUMMARY_FOLDER, 'project_summary.md')
        with open(summary_file, 'w', encoding='utf-8') as file:
            file.write(markdown_content)

        return summary_file


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        session['api_key'] = request.form['api_key']
        session['base_url'] = request.form['base_url']
        session['model'] = request.form['model']
        session['output_language'] = request.form['output_language']
        session['max_tokens'] = int(request.form['max_tokens'])
        session['temperature'] = float(request.form['temperature'])
        session['timeout'] = int(request.form['timeout'])
        session['system_message'] = request.form['system_message']
        session['max_concurrency'] = int(request.form['max_concurrency'])
        return redirect(url_for('index'))

    return render_template(
        'settings.html',
        api_key=session.get('api_key', ''),
        base_url=session.get('base_url', 'https://api.openai.com/v1/'),
        model=session.get('model', 'gpt-4'),
        output_language=session.get('output_language', 'en'),
        max_tokens=session.get('max_tokens', 150),
        temperature=session.get('temperature', 0.7),
        timeout=session.get('timeout', 10),
        system_message=session.get('system_message', 'You are a helpful assistant.'),
        max_concurrency=session.get('max_concurrency', 5),
    )


@app.route('/get_models', methods=['POST'])
def get_models():
    api_key = request.form['api_key']
    base_url = request.form['base_url']
    openai_config = OpenAIConfig(api_key=api_key, base_url=base_url)
    try:
        models = openai_config.get_models()
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'api_key' not in session or 'base_url' not in session:
            return redirect(url_for('settings'))

        api_key = session['api_key']
        base_url = session['base_url']
        model = session.get('model', 'gpt-4')
        output_language = session.get('output_language', 'en')
        max_tokens = session.get('max_tokens', 150)
        temperature = session.get('temperature', 0.7)
        timeout = session.get('timeout', 10)
        system_message = session.get('system_message', 'You are a helpful assistant.')
        max_concurrency = session.get('max_concurrency', 5)
        project_path = request.form.get('project_path')
        project_zip = request.files.get('project_zip')

        if project_zip:
            project_zip_path = os.path.join(UPLOAD_FOLDER, project_zip.filename)
            project_zip.save(project_zip_path)
        else:
            project_zip_path = None

        global analyzing
        analyzing = True
        openai_config = OpenAIConfig(
            api_key=api_key, base_url=base_url, model=model, max_tokens=max_tokens, temperature=temperature, timeout=timeout, system_message=system_message
        )
        analyzer = ProjectAnalyzer(openai_config, project_path, project_zip_path, output_language=output_language)
        thread = threading.Thread(target=analyze_project_thread, args=(analyzer, max_concurrency, request.host_url))
        thread.start()

        return redirect(url_for('progress'))

    return render_template('index.html')


@app.route('/progress')
def progress():
    return render_template('progress.html')


@app.route('/download_summary/<filename>')
def download_summary(filename):
    return send_file(os.path.join(SUMMARY_FOLDER, filename), as_attachment=True)


@app.route('/view_summary/<filename>')
def view_summary(filename):
    with open(os.path.join(SUMMARY_FOLDER, filename), 'r', encoding='utf-8') as file:
        content = file.read()
    html_content = markdown.markdown(content)
    return render_template('view_summary.html', content=Markup(html_content))


@socketio.on('stop_analysis')
def stop_analysis():
    global analyzing
    analyzing = False
    emit('analysis_stopped')


def analyze_project_thread(analyzer, max_concurrency, host_url):
    with app.app_context():
        project_summary, module_summaries = analyzer.analyze_project(max_concurrency)
        if project_summary is None and module_summaries is None:
            return
        summary_file = analyzer.save_summary_as_markdown(project_summary, module_summaries)
        download_url = f"{host_url}download_summary/{os.path.basename(summary_file)}"
        view_url = f"{host_url}view_summary/{os.path.basename(summary_file)}"
        socketio.emit('analysis_complete', {'summary_file': download_url, 'view_file': view_url})


if __name__ == '__main__':
    socketio.run(app, debug=True)
