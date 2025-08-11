import gradio as gr
import tempfile
import os
import json
from doc_utils import build_file_summary, save_json_summary, add_comments_to_docx
from adgm_checker import ADGMChecker
from ragsys_prod_rest import RAGManager, EmbedderREST, VectorStore, check_knowledge_base

# Initialize RAG system with error handling
try:
    embedder = EmbedderREST()
    vector_store = VectorStore()
    rag_system = RAGManager(embedder, vector_store)
    rag_initialized = True
except Exception as e:
    import traceback
    print(f"Error initializing RAG system: {e}")
    print(traceback.format_exc())
    rag_initialized = False
    rag_system = None

def analyze_docx_files(files):
    """
    Analyze DOCX files with ADGM compliance checking and RAG, producing task-compliant outputs.
    """
    if not files:
        return "Please upload files to analyze.", None, None, None
    
    # Check if RAG system is initialized
    if not rag_initialized:
        return "RAG system failed to initialize. Check API key and logs.", None, None, None
        
    # Check if knowledge base exists
    if not check_knowledge_base():
        return "No ADGM knowledge base found. Run ingestion_adgm_source.py.", None, None, None
    
    summaries = []
    all_issues = []
    rag_insights = []
    reviewed_files = []
    html_parts = []
    json_path = None
    # Process each uploaded file and build summaries
    for uploaded in files:
        path = uploaded.name if hasattr(uploaded, "name") else uploaded
        summary = build_file_summary(path, rag_system)
        summaries.append(summary)
        
        # Add the document to reviewed_files list for later processing
        reviewed_files.append(path)

    # Now process each summary for compliance (AI or fallback)
    for summary in summaries:
        rag_response = None
        if summary['detected_type'] != 'Unknown':
            rag_response = rag_system.analyze_document(
                doc_text=summary.get('full_text', summary['preview']),
                doc_type=summary['detected_type']
            )
        rag_success = rag_response and 'error' not in rag_response and (
            ('document_analysis' in rag_response and rag_response.get('compliance_flags')) or
            ('flags' in rag_response and rag_response.get('flags'))
        )
        if rag_success:
            if 'document_analysis' in rag_response:
                rag_insights.append({
                    'document': os.path.basename(summary['filename']),
                    'type': summary['detected_type'],
                    'insights': rag_response['document_analysis'].get('summary', ''),
                    'flags': rag_response.get('compliance_flags', []),
                    'compliance_score': rag_response['document_analysis'].get('compliance_score', 0),
                    'requirements': rag_response.get('adgm_requirements', {}),
                    'recommendations': rag_response.get('recommendations', [])
                })
                for f in rag_response.get('compliance_flags', []):
                    f['document'] = os.path.basename(summary['filename'])
                all_issues.extend(rag_response.get('compliance_flags', []))
            else:
                rag_insights.append({
                    'document': os.path.basename(summary['filename']),
                    'type': summary['detected_type'],
                    'insights': rag_response.get('summary', ''),
                    'flags': rag_response.get('flags', []),
                    'compliance_score': 100 if not rag_response.get('flags', []) else max(0, 100 - 20 * len([i for i in rag_response.get('flags', []) if i.get('severity', '').lower() == 'high']) - 10 * len([i for i in rag_response.get('flags', []) if i.get('severity', '').lower() == 'medium']) - 2 * len(rag_response.get('flags', []))),
                    'requirements': {},
                    'recommendations': rag_response.get('recommendations', [])
                })
                for f in rag_response.get('flags', []):
                    f['document'] = os.path.basename(summary['filename'])
                all_issues.extend(rag_response.get('flags', []))
        else:
            # Fallback to rule-based compliance if AI fails
            from doc_utils import detect_red_flags
            fallback_issues = detect_red_flags(summary.get('full_text', summary['preview']), summary['detected_type'])
            for issue in fallback_issues:
                issue['document'] = os.path.basename(summary['filename'])
            all_issues.extend(fallback_issues)
            rag_insights.append({
                'document': os.path.basename(summary['filename']),
                'type': summary['detected_type'],
                'insights': 'Rule-based compliance analysis used due to AI failure.',
                'flags': fallback_issues,
                'compliance_score': 100 if not fallback_issues else max(0, 100 - 20 * len([i for i in fallback_issues if i['severity'].lower() == 'high']) - 10 * len([i for i in fallback_issues if i['severity'].lower() == 'medium']) - 2 * len(fallback_issues)),
                'requirements': {},
                'recommendations': ['Rule-based recommendations only.']
            })
    process_type = "ADGM Compliance Review"  # or set dynamically as needed
    completeness = "Complete"  # or set dynamically as needed
    structured_report = {
        "process": process_type,
        "completeness": completeness,
        "issues": all_issues,
        "insights": rag_insights,
        "files": summaries
    }
    
    # Save JSON report to a temporary file for download
    json_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
    json.dump(structured_report, json_temp, indent=2, ensure_ascii=False)
    json_temp.close()
    json_path = json_temp.name
    
    if all_issues:
        high_issues = [i for i in all_issues if i['severity'].lower() == 'high']
        medium_issues = [i for i in all_issues if i['severity'].lower() == 'medium']
        low_issues = [i for i in all_issues if i['severity'].lower() == 'low']
        
        # Create detailed issues display
        issues_html = ""
        
        if high_issues:
            issues_html += "<h4 style='color: #dc3545; margin-top: 15px;'>🚨 HIGH SEVERITY FLAGS:</h4>"
            for issue in high_issues[:5]:  # Show first 5 high severity issues
                issues_html += f"""
                <div style="padding: 10px; margin: 5px 0; border-left: 4px solid #dc3545;">
                    <strong>Paragraph:</strong> {issue.get('section', 'N/A')}<br>
                    <strong>Issue:</strong> {issue['issue']}<br>
                    <strong>Reason:</strong> {issue.get('adgm_source', 'ADGM reference')}<br>
                    <strong>Suggestion:</strong> {issue['suggestion']}
                </div>
                """
        
        if medium_issues:
            issues_html += "<h4 style='color: #ffc107; margin-top: 15px;'>🟠 MEDIUM SEVERITY FLAGS:</h4>"
            for issue in medium_issues[:5]:  # Show first 5 medium severity issues
                issues_html += f"""
                <div style="padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; ">
                    <strong>Paragraph:</strong> {issue.get('section', 'N/A')}<br>
                    <strong>Issue:</strong> {issue['issue']}<br>
                    <strong>Reason:</strong> {issue.get('adgm_source', 'ADGM reference')}<br>
                    <strong>Suggestion:</strong> {issue['suggestion']}
                </div>
                """
        
        if low_issues:
            issues_html += "<h4 style='color: #28a745; margin-top: 15px;'>🟡 LOW SEVERITY FLAGS:</h4>"
            for issue in low_issues[:3]:  # Show first 3 low severity issues
                issues_html += f"""
                <div style="padding: 10px; margin: 5px 0; border-left: 4px solid #28a745;">
                    <strong>Paragraph:</strong> {issue.get('section', 'N/A')}<br>
                    <strong>Issue:</strong> {issue['issue']}<br>
                    <strong>Reason:</strong> {issue.get('adgm_source', 'ADGM reference')}<br>
                    <strong>Suggestion:</strong> {issue['suggestion']}
                </div>
                """
        
        html_parts.append(f"""
        <div style="padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h3>🚨 Compliance Issues Found</h3>
            <p><strong>Total Issues:</strong> {len(all_issues)}</p>
            <ul>
                <li><strong>High Severity:</strong> {len(high_issues)}</li>
                <li><strong>Medium Severity:</strong> {len(medium_issues)}</li>
                <li><strong>Low Severity:</strong> {len(low_issues)}</li>
            </ul>
            {issues_html}
        </div>
        """)
    
    for summary in summaries:
        issues = summary.get('issues', [])
        issue_html = ""
        if issues:
            issue_html = "<ul>"
            for issue in issues[:3]:
                severity_color = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}.get(issue['severity'], "#6c757d")
                issue_html += f"""
                <li style="margin-bottom: 5px;">
                    <span style="color: {severity_color}; font-weight: bold;">[{issue['severity']}]</span>
                    {issue['issue']} (Section: {issue.get('section', 'N/A')})
                    <br><small>Suggestion: {issue['suggestion']}</small>
                </li>
                """
            if len(issues) > 3:
                issue_html += f"<li>... and {len(issues) - 3} more issues</li>"
            issue_html += "</ul>"
        
        rag_insight = next((r for r in rag_insights if r['document'] == os.path.basename(summary['filename'])), None)
        rag_section = ""
        if rag_insight:
            rag_section = f"""
            <div style="padding: 10px; margin: 10px 0; border-radius: 5px;">
                <h5>🤖 AI Analysis</h5>
                <p style="font-size: 0.9em;">{rag_insight['insights'][:300]}{'...' if len(rag_insight['insights']) > 300 else ''}</p>
            </div>
            """
        
        html_parts.append(f"""
        <div style="border:1px solid #ddd; padding:15px; margin:8px; border-radius:6px;">
            <h4>{os.path.basename(summary['filename'])}</h4>
            <p><strong>Detected Type:</strong> {summary['detected_type']}</p>
            <p><strong>Paragraphs:</strong> {summary['paragraph_count']} | 
               <strong>Issues:</strong> {len(issues)}</p>
            {rag_section}
            {issue_html if issue_html else "<p style='color: green;'> No issues found</p>"}
            <details>
                <summary>Preview (click to expand)</summary>
                <pre style="white-space:pre-wrap; padding: 10px; border-radius: 4px;">{summary['preview']}</pre>
            </details>
        </div>
        """)
    
    combined_html = "<div>" + "\n".join(html_parts) + "</div>"
    
    # Process documents to add comments
    reviewed_docs = []
    for i, doc_path in enumerate(reviewed_files):
        # Get issues for this document
        doc_name = os.path.basename(doc_path)
        doc_issues = [issue for issue in all_issues if issue.get('document') == doc_name]
        
        # Add comments to document and get the path to the reviewed document
        reviewed_doc_path = add_comments_to_docx(doc_path, doc_issues)
        reviewed_docs.append(reviewed_doc_path)
    
    # Prepare comprehensive insights for the textbox
    insights_text = ""
    if rag_insights:
        for insight in rag_insights:
            insights_text += f"📄 {insight['document']} ({insight['type']})\n"
            insights_text += f"🎯 Compliance Score: {insight.get('compliance_score', 0)}/100\n"
            insights_text += f"📋 Summary: {insight['insights'][:200]}...\n"
            
            # Add recommendations if available
            if insight.get('recommendations'):
                insights_text += "💡 Key Recommendations:\n"
                for rec in insight['recommendations'][:3]:
                    insights_text += f"  • {rec}\n"
            
            insights_text += "\n" + "="*50 + "\n\n"
    
    return (combined_html, json_path, reviewed_docs[0] if reviewed_docs else None,
            insights_text if insights_text else "No AI insights available")

# Create the Gradio interface
with gr.Blocks(title="ADGM Corporate Agent") as demo:
    gr.Markdown("""
    # ADGM Corporate Agent — Intelligent Document Review with RAG
    Upload .docx documents for ADGM compliance review.
    """)
    
    with gr.Row():
        file_input = gr.File(file_count="multiple", label="Upload .docx files", file_types=[".docx"])
        analyze_btn = gr.Button(" Analyze Documents", variant="primary")
    
    with gr.Row():
        output_html = gr.HTML(label="Analysis Results")
    
    with gr.Row():
        json_download = gr.File(label="📄 Download JSON Report")
        reviewed_download = gr.File(label=" Download Reviewed Document")
    
    with gr.Row():
        rag_insights = gr.Textbox(label="🧠 AI Insights", lines=5, interactive=False)
    
    analyze_btn.click(
        fn=analyze_docx_files, 
        inputs=[file_input], 
        outputs=[output_html, json_download, reviewed_download, rag_insights]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        show_api=False,
        max_threads=10
    )
