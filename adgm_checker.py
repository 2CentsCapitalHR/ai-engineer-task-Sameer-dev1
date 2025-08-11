from typing import List, Dict
from ragsys_prod_rest import RAGManager

class ADGMChecker:
    def __init__(self, rag_system: RAGManager = None):
        self.rag_system = rag_system
        self.checklists = {
            "Company Incorporation": [
                "Articles of Association", "Memorandum of Association",
                "Incorporation Application Form", "UBO Declaration Form",
                "Register of Members and Directors"
            ],
            "Licensing Application": [
                "License Application Form", "Business Plan", "Financial Projections"
            ],
            "Compliance Review": [
                "Data Protection Policy", "Employment Contract"
            ],
            "Address Change": [
                "Change of Registered Address Notice"
            ]
        }

    def detect_process_type(self, summaries: List[Dict]) -> str:
        """
        Detect the legal process based on uploaded document types.
        """
        # Extract document types, filtering out unknowns
        doc_types = []
        if summaries:
            for s in summaries:
                if isinstance(s, dict) and s.get('detected_type') and s['detected_type'] != 'Unknown':
                    doc_types.append(s['detected_type'])
                
        if not doc_types:
            self.last_rag_error = "No recognized document types found."
            return "Unknown"
            
        # First try heuristic matching for reliability
        for process, required_docs in self.checklists.items():
            matches = [doc_type for doc_type in doc_types if doc_type in required_docs]
            if matches:
                # If we have a strong match (more than one document matches), return immediately
                if len(matches) > 1:
                    self.last_rag_error = f"Process type detected heuristically with {len(matches)} matching documents."
                    return process
        
        # Use RAG to refine process detection if available
        self.last_rag_error = None
        if self.rag_system:
            try:
                # Create a more descriptive prompt for the RAG system
                doc_type_text = f"Document types: {', '.join(doc_types)}. Determine which ADGM process these documents are for."
                
                # Call the RAG system
                rag_response = self.rag_system.analyze_document(
                    doc_text=doc_type_text,
                    doc_type="Process Detection"
                )
                
                # Check for errors
                if isinstance(rag_response, dict) and 'error' in rag_response:
                    self.last_rag_error = f"RAG error: {rag_response.get('error')}"
                
                # Check for process type in response
                if isinstance(rag_response, dict):
                    # Try different fields where the process might be mentioned
                    response_text = ''
                    if 'summary' in rag_response:
                        response_text = rag_response['summary']
                    elif 'document_analysis' in rag_response and 'summary' in rag_response['document_analysis']:
                        response_text = rag_response['document_analysis']['summary']
                    elif 'comments' in rag_response and rag_response['comments']:
                        response_text = ' '.join([c.get('comment', '') for c in rag_response['comments']])
                    
                    # Check for process types in the response text
                    response_text = response_text.lower()
                    if "company incorporation" in response_text or "incorporation" in response_text:
                        return "Company Incorporation"
                    elif "licensing application" in response_text or "license application" in response_text:
                        return "Licensing Application"
                    elif "compliance review" in response_text or "compliance" in response_text:
                        return "Compliance Review"
                    elif "address change" in response_text or "change of address" in response_text:
                        return "Address Change"
            except Exception as e:
                self.last_rag_error = f"RAG system error: {str(e)}"
        
        # Fallback to basic heuristic if RAG didn't work
        for process, required_docs in self.checklists.items():
            if any(doc_type in required_docs for doc_type in doc_types):
                self.last_rag_error = "Process type detected by basic heuristic matching."
                return process
                
        self.last_rag_error = "Process type could not be detected by AI or heuristic."
        return "Unknown"

    def check_completeness(self, summaries: List[Dict], process_type: str) -> Dict:
        """
        Check if all required documents are present for the process.
        """
        if process_type not in self.checklists:
            return {"uploaded": len(summaries), "required": 0, "missing": []}
        
        required_docs = self.checklists[process_type]
        uploaded_types = [s['detected_type'] for s in summaries if s['detected_type'] != 'Unknown']
        missing = [doc for doc in required_docs if doc not in uploaded_types]
        
        return {
            "uploaded": len(uploaded_types),
            "required": len(required_docs),
            "missing": missing
        }

    def generate_structured_report(self, summaries: List[Dict], process_type: str, completeness: Dict, issues: List[Dict]) -> Dict:
        """
        Generate a structured JSON report per task requirements.
        """
        # Calculate overall compliance score
        total_issues = len(issues)
        high_issues = len([i for i in issues if i.get('severity', '').lower() == 'high'])
        medium_issues = len([i for i in issues if i.get('severity', '').lower() == 'medium'])
        
        # Simple scoring algorithm
        base_score = 100
        score_deduction = (high_issues * 20) + (medium_issues * 10) + (total_issues * 2)
        compliance_score = max(0, base_score - score_deduction)
        
        # Get RAG/LLM error info if present
        rag_error = getattr(self, 'last_rag_error', None)
        
        # Create document summary
        doc_summary = f"Process analysis for {process_type} with {len(summaries)} documents uploaded"
        if rag_error:
            doc_summary += f". Note: {rag_error}"
        
        # Build recommendations based on issues
        recommendations = []
        if high_issues > 0:
            recommendations.append("Address all high severity issues immediately to ensure compliance")
        if medium_issues > 0:
            recommendations.append("Review medium severity issues before submission")
        if completeness['missing']:
            recommendations.append(f"Submit the missing required documents: {', '.join(completeness['missing'])}")
        
        return {
            "document_analysis": {
                "document_type": process_type,
                "compliance_score": compliance_score,
                "total_paragraphs": sum(s.get('paragraph_count', 0) for s in summaries),
                "summary": doc_summary,
                "rag_status": "Success" if not rag_error else "Warning",
                "rag_message": rag_error if rag_error else "Analysis completed successfully"
            },
            "process": process_type,
            "documents_uploaded": len(summaries),
            "required_documents": completeness['required'],
            "missing_document": completeness['missing'],
            "compliance_flags": issues,
            "recommendations": recommendations,
            "adgm_requirements": {
                "missing": completeness['missing'],
                "present": [s.get('detected_type', 'Unknown') for s in summaries if isinstance(s, dict) and s.get('detected_type')]
            }
        }