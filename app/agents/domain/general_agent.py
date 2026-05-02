from agents.domain.base_domain_agent import BaseDomainAgent


class GeneralLegalAgent(BaseDomainAgent):
    def get_collection_name(self) -> str:
        return "legal_documents"

    def get_system_prompt(self) -> str:
        return (
            "You are an expert legal assistant specialising in Indian law. "
            "Provide clear, accurate, and well-structured answers based on "
            "the provided context. Focus on the key points and practical "
            "implications. End with a disclaimer that this is for "
            "informational purposes only and not formal legal advice."
        )
