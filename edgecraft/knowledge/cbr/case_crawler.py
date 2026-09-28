"""Case crawler for building the CBR case store from external sources."""
import json
import re
import uuid
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from edgecraft.core.modality import Modality, TaskType
from edgecraft.utils.llm import create_chat_llm
from .case_store import Case, CaseStore

_CRAWL_SOURCES_PATH = Path(__file__).parent.parent.parent / "config" / "crawl_sources.yaml"


def _load_crawl_sources() -> Dict[str, Any]:
    """Load crawl sources from config/crawl_sources.yaml."""
    if not _CRAWL_SOURCES_PATH.exists():
        logger.warning(f"crawl_sources.yaml not found at {_CRAWL_SOURCES_PATH}, using empty config")
        return {"kaggle": {"keywords": []}, "github": {"repos": []}}
    with open(_CRAWL_SOURCES_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


CASE_PARSING_PROMPT = """You are an expert at extracting structured edge AI case information from notebooks, README files, and project descriptions.

Given the following content from an edge AI project, extract the case information into a JSON object.

**Content:**
{content}

**Source URL:** {source_url}

Extract the following fields:
1. "intent": A natural language description of what this project does (1-2 sentences)
2. "modality": One of [vision, audio, text, time_series, multimodal, structured]
3. "task_type": One of [classification, object_detection, segmentation, pose_estimation, crowd_counting, speech_recognition, audio_classification, text_generation, text_classification, anomaly_detection, regression]
4. "dataset_description": Brief description of the dataset used
5. "target_device": Target deployment device if mentioned (e.g., "jetson_orin_nano", "raspberry_pi_4", "jetson_xavier_nx")
6. "constraints": Dict of performance constraints mentioned (e.g., {{"latency_ms": 50, "memory_mb": 512}})
7. "model_name": The model used (e.g., "yolo11n", "mobilenetv3", "resnet18")
8. "model_source": Where the model comes from (e.g., "ultralytics", "timm", "huggingface", "pytorch")
9. "training_config": Training configuration if mentioned (e.g., {{"epochs": 100, "batch_size": 32, "lr": 0.001}})
10. "optimization_config": Optimization techniques used (e.g., {{"quantization": "int8", "pruning": true, "tensorrt": true}})
11. "metrics": Performance metrics achieved (e.g., {{"accuracy": 0.95, "latency_ms": 45, "mAP": 0.82}})
12. "success": true if the project achieved its goals, false otherwise
13. "lessons_learned": Key insights or lessons from this project

If a field cannot be determined from the content, use reasonable defaults:
- modality: "vision" (most common for edge AI)
- task_type: "classification"
- target_device: ""
- constraints: {{}}
- training_config: {{}}
- optimization_config: {{}}
- metrics: {{}}
- success: true
- lessons_learned: ""

Return ONLY valid JSON, no other text.
"""


class CaseCrawler:
    """Crawler for building the case store from external sources."""

    def __init__(self, llm = None):
        """Initialize the crawler.

        Args:
            llm: Optional LLM for parsing. Uses default if not provided.
        """
        self.llm = llm or create_chat_llm(temperature=0.1, purpose="case_crawler")
        self.case_store = CaseStore()

    def crawl_kaggle(
        self,
        keywords: List[str] = None,
        top_k: int = 50
    ) -> List[Case]:
        """Crawl Kaggle notebooks for edge AI cases.

        Args:
            keywords: Search keywords. Loaded from crawl_sources.yaml if not provided.
            top_k: Maximum notebooks to fetch per keyword.

        Returns:
            List of parsed cases.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            logger.warning("Kaggle API not installed. Run: pip install kaggle")
            return []

        if keywords is None:
            sources = _load_crawl_sources()
            keywords = sources.get("kaggle", {}).get("keywords", [])

        try:
            api = KaggleApi()
            api.authenticate()
        except Exception as e:
            logger.error(f"Kaggle authentication failed: {e}")
            return []

        cases = []
        seen_refs = set()

        for keyword in keywords:
            logger.info(f"Searching Kaggle for: {keyword}")
            try:
                # Kaggle API v2 uses kernels_list() (no kernels_list_with_http_info)
                notebooks = api.kernels_list(
                    search=keyword,
                    sort_by="relevance",
                    language="python",
                    page_size=min(top_k, 20)  # Kaggle limits page size
                ) or []
                notebooks = [nb for nb in notebooks if nb is not None]

                for notebook in notebooks:
                    ref = getattr(notebook, "ref", None) or (notebook.get("ref", "") if isinstance(notebook, dict) else "")
                    if not ref or ref in seen_refs:
                        continue
                    seen_refs.add(ref)

                    try:
                        # Kaggle API v2: kernels_pull(kernel, path) writes to disk
                        with tempfile.TemporaryDirectory() as tmpdir:
                            api.kernels_pull(ref, path=tmpdir, quiet=True)
                            source = None
                            for name in os.listdir(tmpdir):
                                if name.endswith(".ipynb"):
                                    with open(os.path.join(tmpdir, name), "r", encoding="utf-8") as f:
                                        source = f.read()
                                    break
                            if not source:
                                continue

                            nb_data = json.loads(source)
                            cells = nb_data.get("cells", [])
                            content = self._extract_notebook_content(cells)

                        if len(content) < 100:  # Skip too short notebooks
                            continue

                        source_url = f"https://www.kaggle.com/code/{ref}"
                        case = self._parse_content_to_case(
                            content=content[:8000],  # Limit content length
                            source="kaggle",
                            source_url=source_url
                        )

                        if case:
                            cases.append(case)
                            logger.info(f"Parsed case from Kaggle: {case.id}")

                    except Exception as e:
                        logger.debug(f"Failed to process notebook {ref}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Kaggle search failed for '{keyword}': {e}")
                continue

        return cases

    def crawl_github(
        self,
        repos: List[str] = None,
        search_query: str = None,
        min_stars: int = 100,
        top_k: int = 50
    ) -> List[Case]:
        """Crawl GitHub repositories for edge AI cases.

        Args:
            repos: Specific repos to crawl (owner/repo format). Loaded from
                crawl_sources.yaml if not provided.
            search_query: Search query for discovering additional repos.
            min_stars: Minimum stars for searched repos.
            top_k: Maximum repos to process.

        Returns:
            List of parsed cases.
        """
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed")
            return []

        if repos is None:
            sources = _load_crawl_sources()
            repos = sources.get("github", {}).get("repos", [])

        cases = []

        # Process specified repos
        for repo in repos[:top_k]:
            logger.info(f"Processing GitHub repo: {repo}")
            case = self._process_github_repo(repo)
            if case:
                cases.append(case)

        # Search for more repos if query provided
        if search_query:
            try:
                headers = {}
                github_token = getattr(settings, 'GITHUB_TOKEN', None)
                if github_token:
                    headers["Authorization"] = f"token {github_token}"

                response = requests.get(
                    "https://api.github.com/search/repositories",
                    params={
                        "q": f"{search_query} stars:>{min_stars}",
                        "sort": "stars",
                        "per_page": min(top_k, 30)
                    },
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 200:
                    items = response.json().get("items", [])
                    for item in items:
                        repo_name = item.get("full_name", "")
                        if repo_name and repo_name not in repos:
                            case = self._process_github_repo(repo_name)
                            if case:
                                cases.append(case)
            except Exception as e:
                logger.warning(f"GitHub search failed: {e}")

        return cases

    def _process_github_repo(self, repo: str) -> Optional[Case]:
        """Process a single GitHub repository.

        Args:
            repo: Repository in owner/repo format.

        Returns:
            Parsed case or None.
        """
        try:
            import requests
        except ImportError:
            return None

        try:
            # Fetch README
            readme_url = f"https://raw.githubusercontent.com/{repo}/main/README.md"
            response = requests.get(readme_url, timeout=10)

            if response.status_code != 200:
                # Try master branch
                readme_url = f"https://raw.githubusercontent.com/{repo}/master/README.md"
                response = requests.get(readme_url, timeout=10)

            if response.status_code != 200:
                return None

            content = response.text[:8000]  # Limit content
            source_url = f"https://github.com/{repo}"

            return self._parse_content_to_case(
                content=content,
                source="github",
                source_url=source_url
            )

        except Exception as e:
            logger.debug(f"Failed to process GitHub repo {repo}: {e}")
            return None

    def _extract_notebook_content(self, cells: List[Dict]) -> str:
        """Extract readable content from notebook cells.

        Args:
            cells: List of notebook cells.

        Returns:
            Combined text content.
        """
        parts = []
        for cell in cells:
            cell_type = cell.get("cell_type", "")
            source = cell.get("source", "")

            if isinstance(source, list):
                source = "".join(source)

            if cell_type == "markdown":
                parts.append(source)
            elif cell_type == "code":
                # Include code with context
                parts.append(f"```python\n{source}\n```")

        return "\n\n".join(parts)

    def _parse_content_to_case(
        self,
        content: str,
        source: str,
        source_url: str
    ) -> Optional[Case]:
        """Parse content into a Case using LLM.

        Args:
            content: Text content to parse.
            source: Source identifier (kaggle/github/jetson).
            source_url: URL to the original source.

        Returns:
            Parsed Case or None if parsing fails.
        """
        prompt = CASE_PARSING_PROMPT.format(
            content=content,
            source_url=source_url
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert at extracting structured information from technical documents."),
                HumanMessage(content=prompt)
            ])

            # Extract JSON from response
            text = response.content
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())

            # Map string values to enums
            modality_str = data.get("modality", "vision").lower()
            task_type_str = data.get("task_type", "classification").lower()

            try:
                modality = Modality(modality_str)
            except ValueError:
                modality = Modality.VISION

            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.CLASSIFICATION

            # Clean metrics - ensure all values are numeric
            raw_metrics = data.get("metrics", {})
            metrics = {}
            for k, v in raw_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
                elif isinstance(v, str):
                    # Try to extract numeric value from string
                    import re
                    match = re.search(r'[\d.]+', v)
                    if match:
                        try:
                            metrics[k] = float(match.group())
                        except ValueError:
                            pass  # Skip non-numeric metrics

            # Clean constraints - ensure all values are appropriate types
            raw_constraints = data.get("constraints", {})
            constraints = {}
            for k, v in raw_constraints.items():
                if isinstance(v, (int, float, str, bool)):
                    constraints[k] = v

            case = Case(
                id=f"case_{source}_{uuid.uuid4().hex[:8]}",
                intent=data.get("intent", "Edge AI project"),
                modality=modality,
                task_type=task_type,
                dataset_description=data.get("dataset_description", ""),
                target_device=data.get("target_device", ""),
                constraints=constraints,
                model_name=data.get("model_name", ""),
                model_source=data.get("model_source", ""),
                training_config=data.get("training_config", {}),
                optimization_config=data.get("optimization_config", {}),
                metrics=metrics,
                success=data.get("success", True),
                lessons_learned=data.get("lessons_learned", ""),
                source=source,
                source_url=source_url,
                created_at=datetime.now().isoformat()
            )

            return case

        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Failed to parse content from {source_url}: {e}")
            return None

    def build_case_store(
        self,
        limit_per_source: int = 50
    ) -> Dict[str, int]:
        """Build the case store by crawling sources defined in crawl_sources.yaml.

        Args:
            limit_per_source: Max cases per source (Kaggle and GitHub).

        Returns:
            Dict with counts per source and total_added.
        """
        results = {}
        all_cases = []

        logger.info("Crawling Kaggle...")
        kaggle_cases = self.crawl_kaggle(top_k=limit_per_source)
        results["kaggle"] = len(kaggle_cases)
        all_cases.extend(kaggle_cases)

        logger.info("Crawling GitHub...")
        github_cases = self.crawl_github(top_k=limit_per_source)
        results["github"] = len(github_cases)
        all_cases.extend(github_cases)

        # Add all cases to store
        if all_cases:
            added = self.case_store.add_cases_batch(all_cases)
            results["total_added"] = added
        else:
            results["total_added"] = 0

        logger.info(f"Case store build complete: {results}")
        return results
