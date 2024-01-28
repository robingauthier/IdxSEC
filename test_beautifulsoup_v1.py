
def test_1():
    from bs4 import BeautifulSoup
    # Sample HTML document
    html_content = """
    <html>
    <head>
        <title>Sample HTML</title>
    </head>
    <body>
        <div id="main">
            <h1>Title</h1>
            <p>This is a paragraph.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
                <li>Item 3</li>
            </ul>
        </div>
    </body>
    </html>
    """
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    # Find and remove specific elements (e.g., remove all <ul> elements)
    for ul_tag in soup.find_all('ul'):
        ul_tag.decompose()
    # Create a new HTML document with the filtered content
    new_html_content = str(soup)
    print(new_html_content)
    # You can now save or use the new_html_content as needed
    #with open('filtered_document.html', 'w') as file:
    #    file.write(new_html_content)

def test_2():
    from lxml import etree
    # Sample HTML document
    html_content = """
    <html>
    <head>
        <title>Sample HTML</title>
    </head>
    <body>
        <div id="main">
            <h1>Title</h1>
            <p>This is a paragraph.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
                <li>Item 3</li>
            </ul>
        </div>
    </body>
    </html>
    """
    # Parse the HTML content with lxml.etree
    root = etree.fromstring(html_content)
    # Find and remove specific elements (e.g., remove all <ul> elements)
    for ul_element in root.xpath('//ul'):
        ul_element.getparent().remove(ul_element)
    # Serialize the modified tree to get the new HTML content
    new_html_content = etree.tostring(root, pretty_print=True, encoding='unicode')
    print(new_html_content)
    # You can now save or use the new_html_content as needed
    #with open('filtered_document.html', 'w') as file:
    #    file.write(new_html_content)

if __name__=='__main__':
    test_1()