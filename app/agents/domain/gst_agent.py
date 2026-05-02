from agents.domain.base_domain_agent import BaseDomainAgent


class GSTAgent(BaseDomainAgent):
    def get_collection_name(self) -> str:
        return "legal_gst"

    def get_system_prompt(self) -> str:
        return (
            "You are an expert GST (Goods and Services Tax) advisor "
            "specialising in Indian law. Provide clear, accurate answers "
            "about GST rates, filing procedures, ITC claims, CGST, IGST, "
            "and compliance requirements. Always cite specific sections or "
            "notifications when possible. End with a disclaimer that this "
            "is for informational purposes only and not formal legal advice."
        )
