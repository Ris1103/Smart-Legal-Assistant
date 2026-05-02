"""
Contract Agent — extracts parameters from the user query and renders a
Jinja2 contract template.

Supported contract types
------------------------
- nda                 (Non-Disclosure Agreement)
- service_agreement   (Service Agreement)
- employment_agreement(Employment Agreement)
"""
import json
import logging
import pathlib
import re
from datetime import datetime
from typing import Optional

import google.generativeai as genai
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from graph.state import AgentState

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "templates" / "contracts"
)

_SUPPORTED_TYPES = {
    "nda": "nda.j2",
    "non-disclosure": "nda.j2",
    "non disclosure": "nda.j2",
    "service_agreement": "service_agreement.j2",
    "service agreement": "service_agreement.j2",
    "employment_agreement": "employment_agreement.j2",
    "employment agreement": "employment_agreement.j2",
}

_PARAM_EXTRACTION_PROMPT = """\
You are a legal parameter extractor. From the USER REQUEST below, extract
parameters for a {contract_type} contract.

Return ONLY a JSON object with the extracted values. Use null for any
parameter the user did not mention. Do not invent values.

Common parameters to look for (include others if relevant):
- party_a_name, party_a_address
- party_b_name, party_b_address (or provider_name/client_name/employer_name/employee_name)
- effective_date (ISO format if mentioned)
- term / duration
- jurisdiction
- purpose (for NDA)
- fee_amount, fee_currency (for service agreements)
- designation, department, basic_salary, ctc (for employment agreements)

USER REQUEST: {query}
"""


class ContractAgent:
    def __init__(self, generative_model: genai.GenerativeModel):
        self.model = generative_model
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def __call__(self, state: AgentState) -> AgentState:
        query = state["query"]
        logger.info(f"ContractAgent processing: '{query[:80]}'")

        contract_type = self._detect_contract_type(query)
        template_file = _SUPPORTED_TYPES.get(contract_type)

        if not template_file:
            return {
                **state,
                "contract_text": "",
                "summary": (
                    f"I don't have a template for '{contract_type}'. "
                    f"Supported types: NDA, Service Agreement, "
                    f"Employment Agreement."
                ),
                "error": f"Unsupported contract type: {contract_type}",
            }

        params = self._extract_params(query, contract_type)
        contract_text = self._render_template(template_file, params)

        return {
            **state,
            "contract_type": contract_type,
            "contract_params": params,
            "contract_text": contract_text,
            "summary": contract_text,
            "retrieved_docs": [],
            "source_files": [f"Template: {template_file}"],
            "search_type": "local",
            "metadata": {
                **(state.get("metadata") or {}),
                "agent": "ContractAgent",
                "contract_type": contract_type,
                "template": template_file,
                "timestamp": datetime.now().isoformat(),
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_contract_type(self, query: str) -> str:
        q = query.lower()
        for keyword, _ in _SUPPORTED_TYPES.items():
            if keyword in q:
                return keyword
        return "nda"  # default

    def _extract_params(
        self, query: str, contract_type: str
    ) -> dict:
        prompt = _PARAM_EXTRACTION_PROMPT.format(
            contract_type=contract_type, query=query
        )
        try:
            response = self.model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            params = json.loads(raw)
            # Remove null values — Jinja2 defaults handle missing keys
            return {k: v for k, v in params.items() if v is not None}
        except Exception as e:
            logger.warning(f"Parameter extraction failed: {e}. Using defaults.")
            return {}

    def _render_template(self, template_file: str, params: dict) -> str:
        try:
            template = self.jinja_env.get_template(template_file)
            return template.render(**params)
        except TemplateNotFound:
            logger.error(f"Template not found: {template_file}")
            return f"Template '{template_file}' not found."
        except Exception as e:
            logger.error(f"Template render error: {e}")
            return f"Error rendering contract template: {e}"
