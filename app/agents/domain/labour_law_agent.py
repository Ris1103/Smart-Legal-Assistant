from agents.domain.base_domain_agent import BaseDomainAgent


class LabourLawAgent(BaseDomainAgent):
    def get_collection_name(self) -> str:
        return "legal_labour_law"

    def get_system_prompt(self) -> str:
        return (
            "You are an expert Labour Law advisor specialising in Indian "
            "law. Provide clear, accurate answers about the Labour Codes "
            "2020, employee rights, provident fund (PF), ESIC, gratuity, "
            "minimum wages, termination procedures, and workplace "
            "compliance. Always cite the relevant Act or Code. End with a "
            "disclaimer that this is for informational purposes only and "
            "not formal legal advice."
        )
