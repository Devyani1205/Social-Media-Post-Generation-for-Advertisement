# social_media_content_generator.py
# Integrated Social Media Content Generator with Multi-Platform Support

import streamlit as st
import httpx
from typing import List, Optional
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
from agno.tools.firecrawl import FirecrawlTools
from agno.tools.nano_banana import NanoBananaTools

from agno.tools import Toolkit
import os
from agno.models.openai import OpenAIChat


# ============== API KEYS (Configure these) ==============fc-662d3f322e1b447db16b5da82b0ce8bd
#GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "b6")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "uhL1n8c")
UNSPLASH_SECRET_KEY = os.getenv("UNSPLASH_SECRET_KEY", "wJ")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "tvzM")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "skYDsA")
groq_api_key = OPENAI_API_KEY
# ============== CUSTOM UNSPLASH TOOLKIT ==============
class UnsplashTools(Toolkit):
    """Custom toolkit for searching Unsplash stock photos."""
    
    def __init__(self, access_key: str = None, **kwargs):
        self.access_key = access_key or UNSPLASH_ACCESS_KEY
        self.base_url = "https://api.unsplash.com"
        
        tools = [
            self.search_photos,
            self.get_random_photo,
        ]
        super().__init__(name="unsplash_tools", tools=tools, **kwargs)
    
    
    def search_photos(self, query: str, per_page: str | int = 5, orientation: str = "landscape") -> str:
        """
        Search for photos on Unsplash based on a query.
        
        Args:
            query (str): Search term for finding relevant photos
            per_page (int): Number of photos to return (default 5, max 30)
            orientation (str): Photo orientation - 'landscape', 'portrait', or 'squarish'
        
        Returns:
            str: JSON string with photo URLs and descriptions
        """
        try:
            # Convert per_page to int if it's a string
            per_page = int(per_page)
            
            headers = {"Authorization": f"Client-ID {self.access_key}"}
            params = {
                "query": query,
                "per_page": min(per_page, 30),
                "orientation": orientation
            }
            
            response = httpx.get(
                f"{self.base_url}/search/photos",
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for photo in data.get("results", []):
                results.append({
                    "id": photo["id"],
                    "description": photo.get("description") or photo.get("alt_description", "No description"),
                    "url_regular": photo["urls"]["regular"],
                    "url_small": photo["urls"]["small"],
                    "url_thumb": photo["urls"]["thumb"],
                    "photographer": photo["user"]["name"],
                    "download_link": photo["links"]["download"]
                })
            
            return f"Found {len(results)} photos:\n" + "\n".join([
                f"- {r['description']}: {r['url_regular']} (by {r['photographer']})"
                for r in results
            ])
        except Exception as e:
            return f"Error searching Unsplash: {str(e)}"
    
    def get_random_photo(self, query: str = None, orientation: str = "landscape") -> str:
        """
        Get a random photo from Unsplash, optionally filtered by query.
        
        Args:
            query (str): Optional search term to filter random photo
            orientation (str): Photo orientation - 'landscape', 'portrait', or 'squarish'
        
        Returns:
            str: Photo URL and details
        """
        try:
            headers = {"Authorization": f"Client-ID {self.access_key}"}
            params = {"orientation": orientation}
            if query:
                params["query"] = query
            
            response = httpx.get(
                f"{self.base_url}/photos/random",
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            photo = response.json()
            
            return f"Random photo: {photo.get('description') or photo.get('alt_description', 'No description')}\nURL: {photo['urls']['regular']}\nPhotographer: {photo['user']['name']}"
        except Exception as e:
            return f"Error getting random photo: {str(e)}"


# ============== PYDANTIC MODELS ==============
class SocialPost(BaseModel):
    content: str = Field(..., description="Post content")
    character_count: int = Field(..., description="Character count")
    hashtags: List[str] = Field(default=[], description="Hashtags for this post")
    image_suggestion: str = Field(default="", description="Suggested image description")

class PostOutput(BaseModel):
    posts: List[SocialPost] = Field(..., description="List of generated posts")
    platform: str = Field(..., description="Target platform")
    topic_summary: str = Field(..., description="Brief summary of researched topic")

class CompetitorInsight(BaseModel):
    competitor_name: str
    content_strategy: str
    hashtag_patterns: List[str]
    posting_frequency: str
    key_themes: List[str]


# ============== PLATFORM CONFIGURATIONS ==============
PLATFORM_CONFIG = {
    "Twitter": {
        "free_limit": 280,
        "paid_limit": 4000,
        "hashtag_count": (3, 5),
        "tone": "concise, punchy, viral-worthy"
    },
    "LinkedIn": {
        "free_limit": 3000,
        "paid_limit": 3000,
        "hashtag_count": (3, 5),
        "tone": "professional, insightful, thought-leadership"
    },
    "Instagram": {
        "free_limit": 2200,
        "paid_limit": 2200,
        "hashtag_count": (5, 30),
        "tone": "engaging, visual-focused, storytelling"
    }
}

INDUSTRIES = [
    "Technology", "Healthcare", "Finance", "E-commerce", "Education",
    "Real Estate", "Marketing", "Food & Beverage", "Travel", "Fashion",
    "Fitness", "Entertainment", "SaaS", "Manufacturing", "Consulting"
]


# ============== AGENT DEFINITIONS ==============
    
def create_agents(groq_api_key: str, firecrawl_api_key: Optional[str] = None, 
                  google_api_key: Optional[str] = None, unsplash_key: Optional[str] = None,
                  tavily_api_key: Optional[str] = None):
    
    """Create all agents with provided API keys"""
    #meta-llama/llama-guard-4-12b
    
    model = OpenAIChat(id="gpt-4o-mini", api_key =groq_api_key)
    # model = Groq(
    #     id="llama-3.3-70b-versatile",
    #     api_key=groq_api_key
    # )
    
    # 1. Web Search Agent (Base Research)
    web_search_agent = Agent(
        name="Web Search Agent",
        model=model,
        tools=[TavilyTools(api_key=tavily_api_key, search_depth="advanced")],
        instructions=[
            "Search the web for current information on the given topic",
            "Find trending discussions and recent news",
            "Identify key talking points and angles",
            "Focus on industry-specific insights",
        ],
        markdown=True,
    )
    
    # 2. Deep Research Agent
    deep_research_agent = Agent(
        name="Deep Research Agent",
        model=model,
        tools=[TavilyTools(api_key=tavily_api_key, search_depth="advanced")],
        instructions=[
            "Conduct thorough research on the topic",
            "Cross-reference multiple sources for accuracy",
            "Think step-by-step to analyze and synthesize findings",
            "Identify unique angles and insights",
            "Document key statistics and facts with sources",
            "Find industry-specific trends and data",
        ],
        markdown=True,
    )
    
    # 3. Competitor Analysis Agent (with FirecrawlTools)
    competitor_tools = [TavilyTools(api_key=tavily_api_key, search_depth="advanced")]
    if firecrawl_api_key:
        competitor_tools.append(
            FirecrawlTools(
                api_key=firecrawl_api_key,
                enable_scrape=True,
                enable_crawl=True,
                enable_search=True,
                limit=5,
            )
        )
    
    competitor_agent = Agent(
        name="Competitor Analyst",
        model=model,
        tools=competitor_tools,
        instructions=[
            "Analyze competitor social media accounts and content strategies",
            "Identify successful hashtag patterns",
            "Note engagement patterns and content themes",
            "Scrape competitor websites for content insights if URL provided",
            "Map website structures and content strategies",
            "Provide actionable competitive intelligence",
        ],
        markdown=True,
    )
    
    # 4. Content Writer Agent
    content_writer = Agent(
        name="Social Media Content Writer",
        model=model,
        instructions=[
            "Create engaging, platform-optimized social media content",
            "Generate smart, relevant hashtags based on topic and industry",
            "Ensure content fits platform character limits",
            "Use hooks, questions, and calls-to-action",
            "Match the tone to the platform and brand voice",
            "Suggest creative image/visual ideas for each post",
        ],
        markdown=True,
    )
    
    # 5. Image Agent (Unsplash + NanoBanana for AI generation)
    image_tools = []
    if unsplash_key:
        image_tools.append(UnsplashTools(access_key=unsplash_key))
    if google_api_key:
        image_tools.append(NanoBananaTools(api_key=google_api_key))
    
    image_agent = Agent(
        name="Image Agent",
        model=model,
        tools=image_tools if image_tools else [TavilyTools(api_key=tavily_api_key)],
        instructions=[
            "Search for relevant stock photos from Unsplash for social media posts",
            "Generate custom AI images when stock photos don't fit the content needs",
            "Match images to the topic, industry, and platform requirements",
            "Provide multiple image options when possible",
            "Consider platform-specific image dimensions",
        ],
        markdown=True,
    )
    
    return web_search_agent, deep_research_agent, competitor_agent, content_writer, image_agent


# ============== STREAMLIT APP ==============
def main():
    st.set_page_config(
        page_title="SocialGenius - AI Content Generator",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 SocialGenius - AI Social Media Content Generator")
    st.markdown("Generate research-backed social media content with smart hashtags and visuals")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        with st.expander("🔑 API Keys", expanded=False):
            # groq_api_key = st.text_input(
            #     "Groq API Key",
            #     value=GROQ_API_KEY,
            #     type="password"
            # )
            firecrawl_api_key = st.text_input(
                "Firecrawl API Key",
                value=FIRECRAWL_API_KEY,
                type="password",
                help="Required for competitor website scraping"
            )
            google_api_key = st.text_input(
                "Google API Key",
                value=GOOGLE_API_KEY,
                type="password",
                help="Required for AI image generation (NanoBanana)"
            )
            unsplash_key = st.text_input(
                "Unsplash Access Key",
                value=UNSPLASH_ACCESS_KEY,
                type="password",
                help="Required for stock photo search"
            )
            tavily_api_key = st.text_input(
                "Tavily API Key",
                value=TAVILY_API_KEY,
                type="password",
                help="Required for web search"
            )
        
        st.divider()
        
        # Platform Selection
        st.subheader("📲 Platform Settings")
        platform = st.selectbox(
            "Select Platform",
            ["LinkedIn", "Instagram", "Twitter"],
            help="Choose your target social media platform"
        )
        
        # Account Type (for Twitter)
        if platform == "Twitter":
            account_type = st.radio(
                "Account Type",
                ["Free (280 chars)", "Paid (4000 chars)"],
                help="Select your Twitter/X account type"
            )
            char_limit = 500 if "Free" in account_type else 4000
        else:
            char_limit = PLATFORM_CONFIG[platform]["free_limit"]
        
        st.info(f"Character limit: {char_limit}")
        
        # Number of Posts
        num_posts = st.slider(
            "Number of Posts",
            min_value=1,
            max_value=10,
            value=5,
            help="How many posts to generate"
        )
        
        st.divider()
        
        # Features Toggle
        st.subheader("🎯 Features")
        use_deep_research = st.checkbox("Deep Research", value=True)
        use_competitor_analysis = st.checkbox("Competitor Analysis", value=False)
        generate_hashtags = st.checkbox("Smart Hashtags", value=True)
        generate_images = st.checkbox("Generate Image Suggestions", value=True)
        use_ai_images = st.checkbox("AI Image Generation (NanoBanana)", value=False)
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Industry Selection
        industry = st.selectbox(
            "Industry",
            INDUSTRIES,
            help="Select your industry for targeted content"
        )
        
        # Brand Name
        brand_name = st.text_input(
            "Brand Name",
            placeholder="Enter your brand name",
            help="Your company or personal brand name"
        )
        
        # Website URL
        website_url = st.text_input(
            "Website URL (Optional)",
            placeholder="https://yourwebsite.com",
            help="Your website for brand context"
        )
        
        # Topic Input
        topic = st.text_area(
            "Topic for Content",
            placeholder="Enter your topic (e.g., 'AI trends in 2024', 'sustainable fashion tips')",
            height=100
        )
        
        # Custom System Prompt
        with st.expander("✏️ Edit System Prompt (Optional)"):
            default_prompt = f"""You are a social media expert creating viral {platform} content for the {industry} industry.
Your posts should be:
- Engaging and shareable
- Informative yet concise
- Include relevant hashtags ({PLATFORM_CONFIG[platform]['hashtag_count'][0]}-{PLATFORM_CONFIG[platform]['hashtag_count'][1]} per post)
- Have strong hooks and CTAs
- Tone: {PLATFORM_CONFIG[platform]['tone']}
- Brand: {brand_name if brand_name else 'Generic'}"""
            
            custom_prompt = st.text_area(
                "System Prompt",
                value=default_prompt,
                height=200
            )
        
        # Competitor URLs (if enabled)
        competitor_urls = []
        if use_competitor_analysis:
            with st.expander("🔍 Competitor Analysis"):
                competitor_input = st.text_area(
                    "Competitor URLs/Accounts (one per line)",
                    placeholder="https://competitor1.com\n@competitor_handle",
                    height=100
                )
                if competitor_input:
                    competitor_urls = [url.strip() for url in competitor_input.split("\n") if url.strip()]
        
        # Generate Button
        generate_btn = st.button("🚀 Generate Content", type="primary", use_container_width=True)
    
    with col2:
        st.header("📤 Output")
        
        if generate_btn and topic and groq_api_key:
            with st.spinner("🔄 Generating content..."):
                try:
                    # Create agents
                    web_agent, research_agent, competitor_agent, writer_agent, image_agent = create_agents(
                        groq_api_key, 
                        firecrawl_api_key if firecrawl_api_key else None,
                        google_api_key if google_api_key else None,
                        unsplash_key if unsplash_key else None,
                        tavily_api_key if tavily_api_key else None
                    )
                    
                    results = {}
                    
                    # Step 1: Web Search
                    with st.status("🔍 Searching the web...", expanded=True) as status:
                        search_response = web_agent.run(
                            f"Search for latest information about: {topic} in the {industry} industry"
                        )
                        results["web_search"] = search_response.content
                        st.write("✅ Web search complete")
                        status.update(label="Web search complete!", state="complete")
                    
                    # Step 2: Deep Research (if enabled)
                    if use_deep_research:
                        with st.status("🧠 Conducting deep research...", expanded=True) as status:
                            research_prompt = f"""
                            Topic: {topic}
                            Industry: {industry}
                            Brand: {brand_name}
                            
                            Previous findings: {results.get('web_search', '')}
                            
                            Conduct deep research and provide:
                            1. Key facts and statistics
                            2. Trending angles for {platform}
                            3. Unique insights for {industry}
                            4. Potential controversy or debate points
                            5. Industry-specific hashtag trends
                            """
                            research_response = research_agent.run(research_prompt)
                            results["deep_research"] = research_response.content
                            st.write("✅ Deep research complete")
                            status.update(label="Deep research complete!", state="complete")
                    
                    # Step 3: Competitor Analysis (if enabled)
                    if use_competitor_analysis and (competitor_urls or website_url):
                        with st.status("📊 Analyzing competitors...", expanded=True) as status:
                            urls_to_analyze = competitor_urls + ([website_url] if website_url else [])
                            competitor_prompt = f"""
                            Analyze these competitors/websites for topic '{topic}' in {industry}:
                            {chr(10).join(urls_to_analyze)}
                            
                            Identify:
                            1. Their {platform} content strategy
                            2. Popular hashtags they use
                            3. Engagement patterns
                            4. Content themes and formats
                            5. Posting frequency
                            """
                            competitor_response = competitor_agent.run(competitor_prompt)
                            results["competitor_analysis"] = competitor_response.content
                            st.write("✅ Competitor analysis complete")
                            status.update(label="Competitor analysis complete!", state="complete")
                    
                    # Step 4: Generate Posts
                    with st.status("✍️ Writing posts...", expanded=True) as status:
                        hashtag_range = PLATFORM_CONFIG[platform]['hashtag_count']
                        post_prompt = f"""
                        {custom_prompt}
                        
                        Topic: {topic}
                        Platform: {platform}
                        Industry: {industry}
                        Brand: {brand_name}
                        
                        IMPORTANT REQUIREMENTS:
                        - MAXIMUM character count per post: {char_limit} characters
                        - Generate exactly {num_posts} posts
                        - Include {hashtag_range[0]}-{hashtag_range[1]} relevant hashtags per post
                        - Generate Hashtags: {generate_hashtags}
                        
                        Research Findings:
                        {results.get('web_search', '')}
                        
                        {results.get('deep_research', '') if use_deep_research else ''}
                        
                        {results.get('competitor_analysis', '') if use_competitor_analysis else ''}
                        
                        For each post, provide:
                        1. Post content (under {char_limit} characters)
                        2. Hashtags ({hashtag_range[0]}-{hashtag_range[1]} relevant ones)
                        3. Character count
                        4. Suggested image/visual description
                        
                        Format each post as:
                        ---
                        Post [number]:
                        [content]
                        
                        Hashtags: #tag1 #tag2 #tag3
                        Characters: [count]
                        Image Suggestion: [brief description of ideal visual]
                        ---
                        """
                        
                        post_response = writer_agent.run(post_prompt)
                        results["posts"] = post_response.content
                        status.update(label="Posts generated!", state="complete")
                    
                    # Step 5: Generate Images (if enabled)
                    if generate_images:
                        with st.status("🖼️ Finding/generating images...", expanded=True) as status:
                            image_prompt = f"""
                            Find relevant images for {platform} posts about: {topic}
                            Industry: {industry}
                            
                            Search for {num_posts} different stock photos that would work well with social media posts.
                            Focus on: professional, engaging, {PLATFORM_CONFIG[platform]['tone']} imagery.
                            """
                            
                            if use_ai_images and google_api_key:
                                image_prompt += "\n\nAlso generate 1-2 custom AI images for unique visuals."
                            
                            image_response = image_agent.run(image_prompt)
                            results["images"] = image_response.content
                            st.write("✅ Images ready")
                            status.update(label="Images ready!", state="complete")
                    
                    # Display Results
                    st.success("✅ Content generated successfully!")
                    
                    # Show Research Summary
                    if use_deep_research:
                        with st.expander("📚 Research Summary", expanded=False):
                            st.markdown(results.get("deep_research", ""))
                    
                    if use_competitor_analysis and competitor_urls:
                        with st.expander("📊 Competitor Insights", expanded=False):
                            st.markdown(results.get("competitor_analysis", ""))
                    
                    # Show Generated Posts
                    st.subheader(f"📱 Generated {platform} Posts")
                    st.markdown(results.get("posts", ""))
                    
                    # Show Image Suggestions
                    if generate_images:
                        with st.expander("🖼️ Image Suggestions", expanded=True):
                            st.markdown(results.get("images", ""))
                    
                    # Download functionality
                    full_output = f"""
# {platform} Content for {brand_name}
## Topic: {topic}
## Industry: {industry}

{results.get('posts', '')}

---
## Research Summary
{results.get('deep_research', 'N/A')}

---
## Image Suggestions
{results.get('images', 'N/A')}
                    """
                    
                    st.download_button(
                        label="📋 Download All Content",
                        data=full_output,
                        file_name=f"{platform.lower()}_content_{topic[:20].replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)
        
        elif generate_btn and not topic:
            st.warning("Please enter a topic to research")
        elif generate_btn and not groq_api_key:
            st.warning("Please enter your Groq API key")


if __name__ == "__main__":
    main()
