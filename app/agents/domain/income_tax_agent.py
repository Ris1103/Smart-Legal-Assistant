from agents.domain.base_domain_agent import BaseDomainAgent


class IncomeTaxAgent(BaseDomainAgent):
    def get_collection_name(self) -> str:
        return "legal_income_tax"

    def get_system_prompt(self) -> str:
        return (
            "You are an expert Income Tax advisor specialising in Indian "
            "law. Provide clear, accurate answers about income tax slabs, "
            "deductions (Sections 80C, 80D, etc.), TDS, advance tax, "
            "ITR filing, and capital gains. Always cite the relevant "
            "section of the Income Tax Act 1961. End with a disclaimer "
            "that this is for informational purposes only and not formal "
            "legal advice."
        )
