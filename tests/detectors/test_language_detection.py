"""Tests for repository language detection and analysis."""

from src.detectors.language_detector import LanguageDetector


class TestLanguageDetectionBasics:
    """Tests for basic language detection scenarios."""

    @staticmethod
    def test_detect_python_as_main_language():
        """Python is detected as main language when most files are .py."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 1000},
            {"path": "src/utils.py", "size": 1500},
            {"path": "tests/test_main.py", "size": 800},
            {"path": "README.md", "size": 500},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_detect_javascript_as_main_language():
        """JavaScript is detected as main language when most files are .js."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/index.js", "size": 2000},
            {"path": "src/utils.js", "size": 1500},
            {"path": "src/app.jsx", "size": 1000},
            {"path": "package.json", "size": 500},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "JavaScript"

    @staticmethod
    def test_detect_typescript_as_main_language():
        """TypeScript is detected as main language when most files are .ts/.tsx."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.ts", "size": 2000},
            {"path": "src/components/App.tsx", "size": 1500},
            {"path": "src/utils.ts", "size": 1200},
            {"path": "tsconfig.json", "size": 200},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "TypeScript"

    @staticmethod
    def test_detect_java_as_main_language():
        """Java is detected as main language when most files are .java."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/Main.java", "size": 3000},
            {"path": "src/Utils.java", "size": 2000},
            {"path": "src/Agent.java", "size": 2500},
            {"path": "pom.xml", "size": 500},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Java"

    @staticmethod
    def test_detect_go_as_main_language():
        """Go is detected as main language when most files are .go."""
        detector = LanguageDetector()
        file_list = [
            {"path": "cmd/main.go", "size": 1500},
            {"path": "pkg/agent.go", "size": 2000},
            {"path": "pkg/utils.go", "size": 1200},
            {"path": "go.mod", "size": 300},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Go"

    @staticmethod
    def test_detect_csharp_as_main_language():
        """C# is detected as main language when most files are .cs."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/Program.cs", "size": 1500},
            {"path": "src/Agent.cs", "size": 2000},
            {"path": "src/Utils.cs", "size": 1200},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "C#"

    @staticmethod
    def test_detect_ruby_as_main_language():
        """Ruby is detected as main language when most files are .rb."""
        detector = LanguageDetector()
        file_list = [
            {"path": "app/main.rb", "size": 1500},
            {"path": "lib/agent.rb", "size": 1200},
            {"path": "lib/utils.rb", "size": 1000},
            {"path": "Gemfile", "size": 200},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Ruby"


class TestLanguageDetectionWeighting:
    """Tests for language detection with file size weighting."""

    @staticmethod
    def test_language_detection_by_file_count():
        """Language is determined by highest count of language files."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 500},
            {"path": "src/utils.py", "size": 500},
            {"path": "src/test.py", "size": 500},
            {"path": "src/index.js", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_language_detection_by_total_file_size():
        """Language can be weighted by total file size instead of count."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 100},
            {"path": "src/utils.py", "size": 100},
            {"path": "src/large.js", "size": 5000},
        ]
        result = detector.detect_main_language(file_list)
        assert result in ("Python", "JavaScript")

    @staticmethod
    def test_ignore_non_code_files_in_detection():
        """Non-code files (config, docs) are not counted for language detection."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 1000},
            {"path": "README.md", "size": 2000},
            {"path": "LICENSE", "size": 500},
            {"path": "package-lock.json", "size": 3000},
            {"path": "docs/guide.txt", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_ignore_vendor_and_dependencies():
        """Files in vendor/node_modules/dependencies are not counted."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 1000},
            {"path": "src/utils.py", "size": 1000},
            {"path": "node_modules/package/index.js", "size": 10000},
            {"path": "vendor/composer/lib.php", "size": 5000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"


class TestLanguageDetectionMixed:
    """Tests for repositories with multiple programming languages."""

    @staticmethod
    def test_detect_main_language_in_polyglot_repo():
        """Main language is detected when repo has multiple languages."""
        detector = LanguageDetector()
        file_list = [
            {"path": "backend/main.py", "size": 2000},
            {"path": "backend/utils.py", "size": 1500},
            {"path": "backend/agent.py", "size": 1500},
            {"path": "frontend/index.js", "size": 2000},
            {"path": "frontend/app.jsx", "size": 1500},
            {"path": "frontend/utils.js", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result in ("Python", "JavaScript")

    @staticmethod
    def test_multi_language_tie_returns_first_or_none():
        """When languages tie, return first encountered or None."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 1000},
            {"path": "src/index.js", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result in ("Python", "JavaScript", None)

    @staticmethod
    def test_frontend_backend_language_detection():
        """Correctly identifies main language in frontend+backend split repos."""
        detector = LanguageDetector()
        file_list = [
            {"path": "backend/src/main.go", "size": 2000},
            {"path": "backend/src/agent.go", "size": 1500},
            {"path": "backend/src/utils.go", "size": 1000},
            {"path": "frontend/src/index.ts", "size": 1500},
            {"path": "frontend/src/app.tsx", "size": 1200},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Go"


class TestLanguageDetectionEdgeCases:
    """Tests for edge cases and special scenarios."""

    @staticmethod
    def test_empty_repository():
        """Empty repository returns None as main language."""
        detector = LanguageDetector()
        file_list = []
        result = detector.detect_main_language(file_list)
        assert result is None

    @staticmethod
    def test_single_file_repository():
        """Single file repository correctly identifies its language."""
        detector = LanguageDetector()
        file_list = [
            {"path": "main.py", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_only_config_files_repository():
        """Repository with only config/doc files returns None."""
        detector = LanguageDetector()
        file_list = [
            {"path": "README.md", "size": 500},
            {"path": "LICENSE", "size": 300},
            {"path": ".gitignore", "size": 100},
            {"path": "package.json", "size": 200},
        ]
        result = detector.detect_main_language(file_list)
        assert result is None

    @staticmethod
    def test_case_insensitive_extension_detection():
        """File extensions are detected case-insensitively."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.PY", "size": 1000},
            {"path": "src/utils.Py", "size": 1000},
            {"path": "src/test.pY", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_double_extension_files():
        """Files with double extensions (.test.js, .spec.py) are handled correctly."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 1000},
            {"path": "src/utils.py", "size": 1000},
            {"path": "tests/main.test.py", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_hidden_files_ignored():
        """Hidden files (starting with dot) are ignored."""
        detector = LanguageDetector()
        file_list = [
            {"path": ".github/workflows/main.yml", "size": 500},
            {"path": ".src/main.py", "size": 1000},
            {"path": "src/main.py", "size": 1000},
            {"path": "src/utils.py", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_non_code_file_exact_match_in_root():
        """Files matching NON_CODE_FILES exactly (case-insensitive) are skipped."""
        detector = LanguageDetector()
        file_list = [
            {"path": "LICENSE", "size": 500},
            {"path": "license", "size": 500},
            {"path": "README", "size": 500},
            {"path": "readme", "size": 500},
            {"path": "CONTRIBUTING", "size": 500},
            {"path": "contributing", "size": 500},
            {"path": "CHANGELOG", "size": 500},
            {"path": "changelog", "size": 500},
            {"path": "src/main.py", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_non_code_file_extension_match():
        """Files with NON_CODE_FILES extensions are skipped in all directories."""
        detector = LanguageDetector()
        file_list = [
            {"path": "docs/guide.md", "size": 1000},
            {"path": "docs/readme.txt", "size": 1000},
            {"path": "config/app.json", "size": 1000},
            {"path": "config/schema.xml", "size": 1000},
            {"path": "deploy/values.yaml", "size": 1000},
            {"path": "deploy/manifest.yml", "size": 1000},
            {"path": "setup.toml", "size": 1000},
            {"path": "package-lock.json", "size": 1000},
            {"path": "dockerfile", "size": 500},
            {"path": "Dockerfile", "size": 500},
            {"path": ".dockerignore", "size": 100},
            {"path": ".env", "size": 100},
            {"path": "src/main.py", "size": 1000},
            {"path": "src/utils.py", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_vendor_path_case_insensitive_matching():
        """Vendor paths are matched case-insensitively."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 1000},
            {"path": "src/utils.py", "size": 1000},
            {"path": "NODE_MODULES/package/index.js", "size": 5000},
            {"path": "Vendor/composer/lib.php", "size": 3000},
            {"path": ".GIT/config", "size": 100},
            {"path": "DIST/bundle.js", "size": 2000},
            {"path": "BUILD/output.o", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_mixed_vendor_and_config_files():
        """Vendor paths and config files are both properly filtered."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 1000},
            {"path": "src/utils.py", "size": 1000},
            {"path": "src/core.py", "size": 1000},
            {"path": "node_modules/lib/index.js", "size": 10000},
            {"path": "node_modules/lib/README.md", "size": 5000},
            {"path": "dist/bundle.js", "size": 8000},
            {"path": "build/artifact.o", "size": 3000},
            {"path": "package.json", "size": 500},
            {"path": "pyproject.toml", "size": 300},
            {"path": "README.md", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_edge_case_extensions_with_vendor_prefix():
        """Files with language extensions in vendor directories are skipped."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 1000},
            {"path": "src/utils.py", "size": 1000},
            {"path": "node_modules/package/index.js", "size": 5000},
            {"path": "node_modules/package/lib.py", "size": 3000},
            {"path": "vendor/composer/src/Class.php", "size": 2000},
            {"path": "src/parser.js", "size": 800},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_pycache_and_hidden_file_filtering():
        """__pycache__ directory and hidden files (except .gitignore) are filtered."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 1000},
            {"path": "src/utils.py", "size": 1000},
            {"path": "__pycache__/main.cpython-39.pyc", "size": 2000},
            {"path": ".mypy_cache/version.txt", "size": 100},
            {"path": ".pytest_cache/config", "size": 100},
            {"path": ".gitignore", "size": 100},
            {"path": ".env.local", "size": 50},
            {"path": ".secrets", "size": 50},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_multiple_non_code_extensions():
        """Multiple non-code extensions (.lock, .yml, etc.) are all properly filtered."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/main.py", "size": 1000},
            {"path": "src/test.py", "size": 1000},
            {"path": "Gemfile.lock", "size": 5000},
            {"path": "package-lock.json", "size": 10000},
            {"path": "poetry.lock", "size": 8000},
            {"path": "requirements.txt", "size": 500},
            {"path": "config.yaml", "size": 1000},
            {"path": "values.yml", "size": 500},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"


class TestLanguageDetectionIntegration:
    """Integration tests with actual repository structures."""

    @staticmethod
    def test_real_world_python_project_structure():
        """Detect Python in realistic project structure."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/__init__.py", "size": 0},
            {"path": "src/main.py", "size": 2000},
            {"path": "src/scanner/scanner.py", "size": 3000},
            {"path": "src/detectors/pattern_detector.py", "size": 2500},
            {"path": "tests/__init__.py", "size": 0},
            {"path": "tests/test_main.py", "size": 1500},
            {"path": "tests/test_scanner.py", "size": 1800},
            {"path": "README.md", "size": 1000},
            {"path": "pyproject.toml", "size": 500},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"

    @staticmethod
    def test_real_world_javascript_project_structure():
        """Detect JavaScript in realistic project structure."""
        detector = LanguageDetector()
        file_list = [
            {"path": "src/index.js", "size": 1500},
            {"path": "src/components/App.jsx", "size": 2000},
            {"path": "src/utils/helpers.js", "size": 1200},
            {"path": "src/services/api.js", "size": 1800},
            {"path": "tests/unit/test_app.spec.js", "size": 1000},
            {"path": "tests/integration/test_api.spec.js", "size": 1500},
            {"path": "package.json", "size": 600},
            {"path": "package-lock.json", "size": 5000},
            {"path": "README.md", "size": 800},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "JavaScript"

    @staticmethod
    def test_real_world_full_stack_project():
        """Detect main language in full-stack project."""
        detector = LanguageDetector()
        file_list = [
            {"path": "backend/src/main.py", "size": 2000},
            {"path": "backend/src/app.py", "size": 1500},
            {"path": "backend/src/models.py", "size": 1200},
            {"path": "backend/tests/test_app.py", "size": 1000},
            {"path": "backend/migrations/001_init.py", "size": 500},
            {"path": "frontend/src/index.tsx", "size": 1500},
            {"path": "frontend/src/App.tsx", "size": 1200},
            {"path": "frontend/src/utils.ts", "size": 800},
            {"path": "frontend/tests/App.test.tsx", "size": 1000},
            {"path": "docker-compose.yml", "size": 400},
            {"path": "README.md", "size": 1000},
        ]
        result = detector.detect_main_language(file_list)
        assert result == "Python"
