; Main Controller: This module will coordinate the
; operations of the other modules. It will instruct the
; Automation Module to navigate to a profile, then tell the Screenshot
; Module to take a picture, pass that picture to
; the Decision Module to generate a score and make a decision from that.

; Set-up:
; 100% zoom in resolution settings
; Laptop must be plugged in or else the save/file dialog will lag.
; Greenshot: set output location to Screenshots folder
; 			 set capture region to f11
;            Capture --> turn "Show notifications" off
; Capture2Text: unbind Win + R so we can open run dialog
;				turn off show popup window
; In powershell (run as admin):
; 	Set-ExecutionPolicy Unrestricted -Scope LocalMachine
; 	Unblock-File -Path "...\Desktop\GitHub\Operation_Love\open_ai_clip\venv\Scripts\activate.ps1"


; Warnings:
; Running this while having another one running inside Shadow PC will cause the one outside to crash due to clipboard conflicts.
; If AirDroid, tap become long presses where the context menu pops up, click "Hoykeys" in its menu and click "Switch input method"


#include automation.ahk
#include screenshot.ahk
#include decision.ahk

#SingleInstance
#WinActivateForce

CoordMode "Mouse", "Window"

targeted_cities_index := FileRead("targeted_cities_index.txt") - 1  ; index start at 1. Making it "- 1" just makes loop easier to start at the index indicated in the file.

restart_bumble()
{					
	; Airdroid recent tasks button
	; F-keys like F2 shortcut for this stops working after awhile, use clicks
	MouseClick "left", 1600, 40
	sleep 1000
	
	; Airddroid swipe up button
	; 2 possible locations for some reason
	MouseClick "left", 1770, 40
	sleep 500
	MouseClick "left", 470	, 30
	sleep 2000
	
	; Click Bumble
	MouseClick "left", -350, 380
	
	; Airddroid switch input button
	send "^a"
	
	; Click Bumble
	MouseClick "left", -350, 370
	sleep 9000
}

main(dating_app, root_directory)
{	
	; Test
	; restart_bumble()
	
	global targeted_cities_index

	; Total: 1448 cities
	targeted_cities := ["Denver, Colorado", "Glendale, Colorado", "Four Square Mile, Colorado", "Sherrelwood, Colorado", "Aurora, Colorado", "Mountain View, Colorado", "Edgewater, Colorado", "Federal Heights, Colorado", "Cherry Creek, Colorado", "Shaw Heights, Colorado", "Thornton, Colorado", "Centennial, Colorado", "Westminster, Colorado", "Lakewood, Colorado", "Arvada, Colorado", "Highlands Ranch, Colorado", "Stonegate, Colorado", "Sierra Ridge, Colorado", "Boulder, Colorado", "Longmont, Colorado", "Greeley, Colorado", "Fort Collins, Colorado", "Colorado Springs, Colorado", "Cheyenne, Wyoming", "Pueblo, Colorado", "Grand Junction, Colorado", "Santa Fe, New Mexico", "Rio Rancho, New Mexico", "Albuquerque, New Mexico", "Amarillo, Texas", "Provo, Utah", "Orem, Utah", "White City, Utah", "Midvale, Utah", "Taylorsville, Utah", "Salt Lake City, Utah", "West Jordan, Utah", "Kearns, Utah", "West Valley City, Utah", "Ogden, Utah", "Logan, Utah", "Pierre, South Dakota", "Wichita, Kansas", "Oaklawn-Sunview, Kansas", "Lincoln, Nebraska", "Idaho Falls, Idaho", "Billings, Montana", "Lubbock, Texas", "Omaha, Nebraska", "Sioux City, Iowa", "Hartford, South Dakota", "Topeka, Kansas", "Sioux Falls, South Dakota", "Oklahoma City, Oklahoma", "Smith Village, Oklahoma", "St. George, Utah", "Norman, Oklahoma", "Las Cruces, New Mexico", "Tortugas, New Mexico", "Bismarck, North Dakota", "Wichita Falls, Texas", "Olathe, Kansas", "Kansas City, Kansas", "Overland Park, Kansas", "Tulsa, Oklahoma", "El Paso, Texas", "Kansas City, Missouri", "Midland, Texas", "Broken Arrow, Oklahoma", "Odessa, Texas", "Independence, Missouri", "Lee's Summit, Missouri", "Scottsdale, Arizona", "Abilene, Texas", "Mesa, Arizona", "Peoria, Arizona", "Phoenix, Arizona", "Gilbert, Arizona", "San Tan Valley, Arizona", "Tempe, Arizona", "Glendale, Arizona", "Guadalupe, Arizona", "Chandler, Arizona", "Surprise, Arizona", "Helena, Montana", "Des Moines, Iowa", "North Las Vegas, Nevada", "Sunrise Manor, Nevada", "Whitney, Nevada", "Henderson, Nevada", "Winchester, Nevada", "Paradise, Nevada", "Las Vegas, Nevada", "Spring Valley, Nevada", "Enterprise, Nevada", "Tucson, Arizona", "Denton, Texas", "San Angelo, Texas", "Paloma Creek, Texas", "Paloma Creek South, Texas", "Savannah, Texas", "Blue Mound, Texas", "Fargo, North Dakota", "Watauga, Texas", "Fayetteville, Arkansas", "Lewisville, Texas", "Fort Worth, Texas", "Frisco, Texas", "McKinney, Texas", "Hebron, Texas", "Boise, Idaho", "Carrollton, Texas", "Allen, Texas", "Plano, Texas", "Irving, Texas", "Arlington, Texas", "Fort Smith, Arkansas", "Meridian, Idaho", "Springfield, Missouri", "Richardson, Texas", "University Park, Texas", "Grand Prairie, Texas", "Cockrell Hill, Texas", "Garland, Texas", "Nampa, Idaho", "Dallas, Texas", "Mesquite, Texas", "Travis Ranch, Texas", "Columbia, Missouri", "St. Cloud, Minnesota", "Spring Park, Minnesota", "Waterloo, Iowa", "Jefferson City, Missouri", "Richfield, Minnesota", "Minneapolis, Minnesota", "Columbia Heights, Minnesota", "Hilltop, Minnesota", "Lauderdale, Minnesota", "St. Paul, Minnesota", "Rochester, Minnesota", "Landfall, Minnesota", "Cedar Rapids, Iowa", "University Heights, Iowa", "Waco, Texas", "Iowa City, Iowa", "Killeen, Texas", "Avenue B and C, Arizona", "Temple, Texas", "Yuma, Arizona", "Tyler, Texas", "Longview, Texas", "Indio, California", "Round Rock, Texas", "Davenport, Iowa", "Wells Branch, Texas", "Heber, California", "Austin, Texas", "Eau Claire, Wisconsin", "Little Rock, Arkansas", "Victorville, California", "Breckenridge Hills, Missouri", "University City, Missouri", "Clayton, Missouri", "Marlborough, Missouri", "Maplewood, Missouri", "Lakeshire, Missouri", "Pasadena Park, Missouri", "Norwood Court, Missouri", "Beverly Hills, Missouri", "Velda Village Hills, Missouri", "Hillsdale, Missouri", "Northwoods, Missouri", "Wilbur Park, Missouri", "St. George, Missouri", "Flordell Hills, Missouri", "San Bernardino, California", "St. Louis, Missouri", "Glasgow Village, Missouri", "Hemet, California", "Sparks, Nevada", "Moreno Valley, California", "Rialto, California", "Carson City, Nevada", "San Antonio, Texas", "College Station, Texas", "Shreveport, Louisiana", "Fontana, California", "Reno, Nevada", "Menifee, California", "Riverside, California", "Rancho Cucamonga, California", "Jurupa Valley, California", "Duluth, Minnesota", "Murrieta, California", "Temecula, California", "Ontario, California", "Home Gardens, California", "Eastvale, California", "Peoria, Illinois", "Coeur d'Alene, Idaho", "Springfield, Illinois", "Montclair, California", "Grandview, Illinois", "Palmdale, California", "Lancaster, California", "Corona, California", "Coronita, California", "Pomona, California", "Desert View Highlands, California", "Orange Cove, California", "Charter Oak, California", "Escondido, California", "Citrus, California", "Covina, California", "Vincent, California", "West Covina, California", "Visalia, California", "South San Jose Hills, California", "Valinda, California", "South Monrovia Island, California", "Baldwin Park, California", "Vista, California", "Spokane Valley, Washington", "Mayflower Village, California", "La Puente, California", "West Puente Valley, California", "North El Monte, California", "Placentia, California", "Oceanside, California", "Bostonia, California", "Mission Viejo, California", "Avocado Heights, California", "El Monte, California", "Orange, California", "Temple City, California", "Anaheim, California", "Parlier, California", "El Cajon, California", "East San Gabriel, California", "San Pasqual, California", "East Niles, California", "La Habra, California", "Carlsbad, California", "South El Monte, California", "Fullerton, California", "Pasadena, California", "Tustin, California", "San Gabriel, California", "Irvine, California", "Rosemead, California", "Hillcrest, California", "Clovis, California", "Rose Hills, California", "Laguna Woods, California", "Whittier, California", "East Whittier, California", "East Bakersfield, California", "Potomac Park, California", "La Cresta, California", "South San Gabriel, California", "La Crescenta-Montrose, California", "Aliso Viejo, California", "La Mesa, California", "Spokane, Washington", "Alhambra, California", "South Pasadena, California", "Cottonwood, California", "South Whittier, California", "La Mirada, California", "Santa Ana, California", "West Whittier-Los Nietos, California", "Monterey Park, California", "Pico Rivera, California", "Montebello, California", "La Presa, California", "Glendale, California", "San Diego, California", "Buena Park, California", "Lemon Grove, California", "Rexland Acres, California", "Benton Park, California", "Bakersfield, California", "Garden Grove, California", "Mayfair, California", "East Los Angeles, California", "Old Stine, California", "Stanton, California", "Norwalk, California", "La Palma, California", "Madison, Wisconsin", "Cerritos, California", "Fresno, California", "Costa Mesa, California", "Fountain Valley, California", "Downey, California", "Cypress, California", "Bell Gardens, California", "Artesia, California", "Midway City, California", "Burbank, California", "Westminster, California", "Santa Clarita, California", "Chula Vista, California", "Bell, California", "Maywood, California", "Hawaiian Gardens, California", "Cudahy, California", "Bellflower, California", "San Fernando, California", "Huntington Park, California", "South Gate, California", "Huntington Beach, California", "National City, California", "Lakewood, California", "Rossmoor, California", "Paramount, California", "Walnut Park, California", "Lynwood, California", "Florence-Graham, California", "East Rancho Dominguez, California", "Willowbrook, California", "West Hollywood, California", "Compton, California", "Signal Hill, California", "Bloomington, Illinois", "Long Beach, California", "Los Angeles, California", "Rockford, Illinois", "West Rancho Dominguez, California", "Imperial Beach, California", "Beverly Hills, California", "Westmont, California", "View Park-Windsor Hills, California", "West Athens, California", "Inglewood, California", "Gardena, California", "Culver City, California", "The Woodlands, Texas", "Lennox, California", "Hawthorne, California", "Alondra Park, California", "West Carson, California", "Kennewick, Washington", "Del Aire, California", "Lawndale, California", "Santa Monica, California", "Marina del Rey, California", "Torrance, California", "Lomita, California", "Manhattan Beach, California", "Redondo Beach, California", "Hermosa Beach, California", "Simi Valley, California", "Merced, California", "Taft Heights, California", "Thousand Oaks, California", "Casa Conejo, California", "Santa Paula, California", "Mission Bend, Texas", "Monroe, Louisiana", "Houston, Texas", "Sugar Land, Texas", "West University Place, Texas", "Southside Place, Texas", "Boulder Hill, Illinois", "San Buenaventura, California", "Riverbank, California", "Oxnard, California", "Memphis, Tennessee", "Elgin, Illinois", "Aurora, Illinois", "Cloverleaf, Texas", "Champaign, Illinois", "Airport, California", "Channel Islands Beach, California", "Bystrom, California", "Citrus Heights, California", "Modesto, California", "Roseville, California", "Rouse, California", "Bret Harte, California", "South Houston, Texas", "Foothill Farms, California", "Salida, California", "Naperville, Illinois", "Joliet, Illinois", "Antelope, California", "North Highlands, California", "La Riviera, California", "Rosemont, California", "Pearland, Texas", "Hanover Park, Illinois", "Pasadena, Texas", "Arden-Arcade, California", "Newman, California", "August, California", "Elk Grove, California", "Florin, California", "Glendale Heights, Illinois", "Ridgewood, Illinois", "Bonnie Brae, Illinois", "Fruitridge Pocket, California", "Stockton, California", "Parkway, California", "Sacramento, California", "Lemon Hill, California", "Round Lake Beach, Illinois", "Lincoln Village, California", "Bend, Oregon", "Laredo, Texas", "Yuba City, California", "Santa Barbara, California", "Indian Creek, Illinois", "Mount Prospect, Illinois", "League City, Texas", "Chico, California", "La Grange, Illinois", "Stone Park, Illinois", "La Grange Park, Illinois", "Bellwood, Illinois", "Arbury Hills, Illinois", "Appleton, Wisconsin", "West Allis, Wisconsin", "Beaumont, Texas", "Orland Hills, Illinois", "Melrose Park, Illinois", "Brookfield, Illinois", "Maywood, Illinois", "Hickory Hills, Illinois", "Park City, Illinois", "Park Ridge, Illinois", "Summit, Illinois", "University of California-Santa Barbara, California", "Forest Park, Illinois", "Norridge, Illinois", "Elmwood Park, Illinois", "Isla Vista, California", "Harwood Heights, Illinois", "Niles, Illinois", "Berwyn, Illinois", "Milwaukee, Wisconsin", "Chicago Ridge, Illinois", "Kenosha, Wisconsin", "Oak Park, Illinois", "Highwood, Illinois", "Burbank, Illinois", "Oak Lawn, Illinois", "Cicero, Illinois", "Davis, California", "Hometown, Illinois", "Rio Bravo, Texas", "Skokie, Illinois", "Whitefish Bay, Wisconsin", "Evergreen Park, Illinois", "Shorewood, Wisconsin", "Racine, Wisconsin", "Blue Island, Illinois", "Chicago, Illinois", "Hamilton City, California", "Evanston, Illinois", "Calumet Park, Illinois", "Santa Maria, California", "Hollister, California", "Yakima, Washington", "Antioch, California", "Galveston, Texas", "Grover Beach, California", "Greenfield, California", "Port Arthur, Texas", "Vacaville, California", "Guadalupe, California", "University, Mississippi", "Soledad, California", "Green Bay, Wisconsin", "Gilroy, California", "Redding, California", "Suisun City, California", "Fairfield, California", "Concord, California", "Alum Rock, California", "Lake Charles, Louisiana", "Milpitas, California", "San Jose, California", "Pacheco, California", "Contra Costa Centre, California", "Corpus Christi, Texas", "Salinas, California", "Saranap, California", "Fremont, California", "Watsonville, California", "Cambrian Park, California", "Santa Clara, California", "Castroville, California", "Campbell, California", "Hayward, California", "Cherryland, California", "Vallejo, California", "Ashland, California", "Sunnyvale, California", "Evansville, Indiana", "San Lorenzo, California", "San Leandro, California", "Cupertino, California", "Mountain View, California", "Oakland, California", "Piedmont, California", "Capitola, California", "East Palo Alto, California", "Seaside, California", "Kensington, California", "Berkeley, California", "Live Oak, California", "Pleasure Point, California", "Tara Hills, California", "El Cerrito, California", "East Richmond Heights, California", "Montalvin Manor, California", "Albany, California", "Twin Lakes, California", "Alameda, California", "Rollingwood, California", "Emeryville, California", "Stanford, California", "San Pablo, California", "Richmond, California", "North Fair Oaks, California", "Santa Cruz, California", "West Menlo Park, California", "Foster City, California", "Boyes Hot Springs, California", "Pacific Grove, California", "San Carlos, California", "White City, Oregon", "Belmont, California", "San Mateo, California", "Medford, Oregon", "Lafayette, Indiana", "Burlingame, California", "San Francisco, California", "South San Francisco, California", "Millbrae, California", "San Bruno, California", "Alto, California", "Marin City, California", "Daly City, California", "Broadmoor, California", "Santa Rosa, California", "Rohnert Park, California", "Jackson, Mississippi", "Gresham, Oregon", "Clarksville, Tennessee", "Lafayette, Louisiana", "Bloomington, Indiana", "Johnson City, Oregon", "Maywood Park, Oregon", "Eugene, Oregon", "South Bend, Indiana", "Orchards, Washington", "Notre Dame, Indiana", "Vancouver, Washington", "Portland, Oregon", "Minnehaha, Washington", "Las Lomas, Texas", "Gervais, Oregon", "Four Corners, Oregon", "Hayesville, Oregon", "King City, Oregon", "Salem, Oregon", "Muskegon, Michigan", "Cedar Mill, Oregon", "Keizer, Oregon", "Marlene Village, Oregon", "Indianapolis, Indiana", "Oak Hills, Oregon", "Bethany, Oregon", "Aloha, Oregon", "Holland, Michigan", "Elkhart, Indiana", "Hillsboro, Oregon", "Edinburg, Texas", "Cornelius, Oregon", "Baton Rouge, Louisiana", "McAllen, Texas", "Kent, Washington", "Renton, Washington", "Nashville, Tennessee", "Bellevue, Washington", "Bryn Mawr-Skyway, Washington", "Tacoma, Washington", "Kalamazoo, Michigan", "Grand Rapids, Michigan", "White Center, Washington", "Harlingen, Texas", "Bothell East, Washington", "Seattle, Washington", "Mill Creek East, Washington", "Alderwood Manor, Washington", "Everett, Washington", "Mountlake Terrace, Washington", "Lake Stickney, Washington", "North Lynnwood, Washington", "Esperance, Washington", "Marysville, Washington", "Olympia, Washington", "Parkway Village, Kentucky", "Strathmoor Manor, Kentucky", "Strathmoor Village, Kentucky", "Norbourne Estates, Kentucky", "Meadowview Estates, Kentucky", "Louisville, Kentucky", "Bremerton, Washington", "Brownsville, Texas", "Cameron Park, Texas", "Fort Wayne, Indiana", "Blue Ridge Manor, Kentucky", "Sycamore, Kentucky", "Fincastle, Kentucky", "Murfreesboro, Tennessee", "Coldstream, Kentucky", "Worthington Hills, Kentucky", "Mandeville, Louisiana", "Huntsville, Alabama", "Tuscaloosa, Alabama", "Houma, Louisiana", "Bellingham, Washington", "Metairie, Louisiana", "Lansing, Michigan", "New Orleans, Louisiana", "Frankfort, Kentucky", "Terrytown, Louisiana", "Timberlane, Louisiana", "Cheviot, Ohio", "Northbrook, Ohio", "North College Hill, Ohio", "Cincinnati, Ohio", "Elmwood Place, Ohio", "Bellevue, Kentucky", "Birmingham, Alabama", "Norwood, Ohio", "Golf Manor, Ohio", "Deer Park, Ohio", "Madison Place, Ohio", "Dayton, Ohio", "Gulfport, Mississippi", "Lexington, Kentucky", "Saginaw, Michigan", "Ann Arbor, Michigan", "South Lyon, Michigan", "Flint, Michigan", "Toledo, Ohio", "Chattanooga, Tennessee", "Mobile, Alabama", "Keego Harbor, Michigan", "Dearborn Heights, Michigan", "Dearborn, Michigan", "Lincoln Park, Michigan", "Berkley, Michigan", "Oak Park, Michigan", "Detroit, Michigan", "Hazel Park, Michigan", "Hamtramck, Michigan", "Warren, Michigan", "Sterling Heights, Michigan", "Lincoln Village, Ohio", "Eastpointe, Michigan", "Grosse Pointe Park, Michigan", "Harper Woods, Michigan", "Montgomery, Alabama", "Grosse Pointe, Michigan", "Grandview Heights, Ohio", "Columbus, Ohio", "Bexley, Ohio", "Knoxville, Tennessee", "Pensacola, Florida", "Auburn, Alabama", "Kennesaw State University, Georgia", "Lorain, Ohio", "South Fulton, Georgia", "Sandy Springs, Georgia", "Atlanta, Georgia", "Huntington, West Virginia", "Decatur, Georgia", "Clarkston, Georgia", "Lakewood, Ohio", "Gainesville, Georgia", "Columbus, Georgia", "Cleveland, Ohio", "Lakeview Estates, Georgia", "Cleveland Heights, Ohio", "University Heights, Ohio", "Akron, Ohio", "Willowick, Ohio", "Kingsport, Tennessee", "Canton, Ohio", "Johnson City, Tennessee", "Charleston, West Virginia", "Athens, Georgia", "Asheville, North Carolina", "Macon, Georgia", "Panama City, Florida", "Youngstown, Ohio", "Warner Robins, Georgia", "Greenville, South Carolina", "Mauldin, South Carolina", "New Brighton, Pennsylvania", "Rochester, Pennsylvania", "Spartanburg, South Carolina", "Erie, Pennsylvania", "Wesleyville, Pennsylvania", "Avalon, Pennsylvania", "Crafton, Pennsylvania", "Ingram, Pennsylvania", "McKees Rocks, Pennsylvania", "Bellevue, Pennsylvania", "West View, Pennsylvania", "Dormont, Pennsylvania", "Castle Shannon, Pennsylvania", "Hickory, North Carolina", "Mount Oliver, Pennsylvania", "Millvale, Pennsylvania", "Pittsburgh, Pennsylvania", "Brentwood, Pennsylvania", "Sharpsburg, Pennsylvania", "Aspinwall, Pennsylvania", "Swissvale, Pennsylvania", "Edgewood, Pennsylvania", "Wilkinsburg, Pennsylvania", "Charleroi, Pennsylvania", "Verona, Pennsylvania", "Turtle Creek, Pennsylvania", "Arnold, Pennsylvania", "Pitcairn, Pennsylvania", "Brackenridge, Pennsylvania", "Tallahassee, Florida", "Gastonia, North Carolina", "Augusta, Georgia", "Rock Hill, South Carolina", "Charlotte, North Carolina", "Roanoke, Virginia", "Indiana, Pennsylvania", "Concord, North Carolina", "Winston-Salem, North Carolina", "Kenmore, New York", "Buffalo, New York", "Eggertsville, New York", "Dale, Pennsylvania", "High Point, North Carolina", "Columbia, South Carolina", "Greensboro, North Carolina", "Lynchburg, Virginia", "Burlington, North Carolina", "University of Virginia, Virginia", "Charlottesville, Virginia", "State College, Pennsylvania", "Savannah, Georgia", "Rochester, New York", "Saint John Fisher College, New York", "Durham, North Carolina", "Hagerstown, Maryland", "Cary, North Carolina", "Shippensburg University, Pennsylvania", "Gainesville, Florida", "Fayetteville, North Carolina", "Raleigh, North Carolina", "North Charleston, South Carolina", "Stone Ridge, Virginia", "Frederick, Maryland", "Jacksonville, Florida", "Bull Run, Virginia", "Sudley, Virginia", "Charleston, South Carolina", "Loch Lomond, Virginia", "Sterling, Virginia", "Hutchison, Virginia", "McNair, Virginia", "Centreville, Virginia", "Manassas Park, Virginia", "Sugarland Run, Virginia", "Herndon, Virginia", "Fair Oaks, Virginia", "Germantown, Maryland", "Fredericksburg, Virginia", "Lewisburg, Pennsylvania", "Dale City, Virginia", "Burke Centre, Virginia", "Gaithersburg, Maryland", "Montgomery Village, Maryland", "Flower Hill, Maryland", "Shiremanstown, Pennsylvania", "Merrifield, Virginia", "Tysons, Virginia", "McSherrystown, Pennsylvania", "Midway, Pennsylvania", "Occoquan, Virginia", "Idylwood, Virginia", "Harrisburg, Pennsylvania", "West Falls Church, Virginia", "Annandale, Virginia", "Falls Church, Virginia", "North Bethesda, Maryland", "Penbrook, Pennsylvania", "Seven Corners, Virginia", "Aspen Hill, Maryland", "SUNY Oswego, New York", "Ocala, Florida", "Leisure World, Maryland", "Bailey's Crossroads, Virginia", "Enhaut, Pennsylvania", "Somerset, Maryland", "Kingstowne, Virginia", "North Kensington, Maryland", "Friendship Heights Village, Maryland", "Arlington, Virginia", "Wheaton, Maryland", "Chevy Chase Section Three, Maryland", "Martin's Additions, Maryland", "Glenmont, Maryland", "Richmond, Virginia", "Forest Glen, Maryland", "Woodlawn, Virginia", "Alexandria, Virginia", "Rutherford, Pennsylvania", "Ithaca, New York", "Kemp Mill, Maryland", "Silver Spring, Maryland", "Huntington, Virginia", "Hybla Valley, Virginia", "Four Corners, Maryland", "Burnt Mills, Maryland", "Takoma Park, Maryland", "Washington, District of Columbia", "Langley Park, Maryland", "West York, Pennsylvania", "Fairland, Maryland", "Chillum, Maryland", "Adelphi, Maryland", "Mount Rainier, Maryland", "York, Pennsylvania", "Hyattsville, Maryland", "Brentwood, Maryland", "University Park, Maryland", "Glassmanor, Maryland", "North Brentwood, Maryland", "Cottage City, Maryland", "College Park, Maryland", "Hillcrest Heights, Maryland", "Reisterstown, Maryland", "Shamokin, Pennsylvania", "Bladensburg, Maryland", "Spring Hill, Florida", "Jasmine Estates, Florida", "Columbia, Maryland", "Temple Hills, Maryland", "East Riverdale, Maryland", "Coral Hills, Maryland", "Laurel, Maryland", "Suitland, Maryland", "Cedar Heights, Maryland", "Landover Hills, Maryland", "Seat Pleasant, Maryland", "New Carrollton, Maryland", "Landover, Maryland", "Peppermill Village, Maryland", "District Heights, Maryland", "Seabrook, Maryland", "Dallastown, Pennsylvania", "Myrtle Beach, South Carolina", "Glenarden, Maryland", "Waldorf, Maryland", "Mount Carmel, Pennsylvania", "Belleair Bluffs, Florida", "Clearwater, Florida", "Syracuse, New York", "Lebanon, Pennsylvania", "Baltimore, Maryland", "Redington Shores, Florida", "North Redington Beach, Florida", "Parkville, Maryland", "South Highpoint, Florida", "Minersville, Pennsylvania", "Frackville, Pennsylvania", "Leesburg, Florida", "Kenneth City, Florida", "Lancaster, Pennsylvania", "Lealman, Florida", "Egypt Lake-Leto, Florida", "South Pasadena, Florida", "Bear Creek, Florida", "Mahanoy City, Pennsylvania", "University, Florida", "Annapolis, Maryland", "Binghamton University, New York", "St. Petersburg, Florida", "North Beach, Maryland", "Tampa, Florida", "Binghamton, New York", "Plymouth, Pennsylvania", "Kingston, Pennsylvania", "Wilkes-Barre, Pennsylvania", "Freeland, Pennsylvania", "West Lawn, Pennsylvania", "Lincoln Park, Pennsylvania", "Brandon, Florida", "Shillington, Pennsylvania", "West Pittston, Pennsylvania", "West Reading, Pennsylvania", "Greenville, North Carolina", "Reading, Pennsylvania", "Laureldale, Pennsylvania", "Mount Penn, Pennsylvania", "Riverview, Florida", "South Bradenton, Florida", "Scranton, Pennsylvania", "West Samoset, Florida", "Bayshore Gardens, Florida", "Kutztown University, Pennsylvania", "Wilmington, North Carolina", "Lakeland, Florida", "Coatesville, Pennsylvania", "Deltona, Florida", "Pine Hills, Florida", "Daytona Beach Shores, Florida", "Newport News, Virginia", "Boyertown, Pennsylvania", "Kennett Square, Pennsylvania", "Tangelo Park, Florida", "Oak Ridge, Florida", "Lake Sarasota, Florida", "Jacksonville, North Carolina", "Coplay, Pennsylvania", "Bethel Manor, Virginia", "Sky Lake, Florida", "Utica, New York", "West Chester, Pennsylvania", "Azalea Park, Florida", "Allentown, Pennsylvania", "East Greenville, Pennsylvania", "Orlando, Florida", "Royersford, Pennsylvania", "Winter Haven, Florida", "Phoenixville, Pennsylvania", "Hampton, Virginia", "Elsmere, Delaware", "Kissimmee, Florida", "Fountain Hill, Pennsylvania", "Wilmington, Delaware", "Buenaventura Lakes, Florida", "Penns Grove, New Jersey", "Souderton, Pennsylvania", "Norfolk, Virginia", "Media, Pennsylvania", "Wilson, Pennsylvania", "Bridgeport, Pennsylvania", "Norristown, Pennsylvania", "West Easton, Pennsylvania", "Easton, Pennsylvania", "Parkside, Pennsylvania", "Chesapeake, Virginia", "Chester, Pennsylvania", "Lansdale, Pennsylvania", "Dover, Delaware", "Bryn Mawr, Pennsylvania", "Conshohocken, Pennsylvania", "Woodlyn, Pennsylvania", "North Wales, Pennsylvania", "Morton, Pennsylvania", "Haverford College, Pennsylvania", "Rutledge, Pennsylvania", "Folsom, Pennsylvania", "Ridley Park, Pennsylvania", "Ardmore, Pennsylvania", "Drexel Hill, Pennsylvania", "Prospect Park, Pennsylvania", "Clifton Heights, Pennsylvania", "Norwood, Pennsylvania", "Aldan, Pennsylvania", "Glenolden, Pennsylvania", "Penn Wynne, Pennsylvania", "Narberth, Pennsylvania", "Lansdowne, Pennsylvania", "Collingdale, Pennsylvania", "Folcroft, Pennsylvania", "Ambler, Pennsylvania", "East Lansdowne, Pennsylvania", "Sharon Hill, Pennsylvania", "Millbourne, Pennsylvania", "Darby, Pennsylvania", "Yeadon, Pennsylvania", "Colwyn, Pennsylvania", "Arcadia University, Pennsylvania", "Glenside, Pennsylvania", "Roslyn, Pennsylvania", "Jenkintown, Pennsylvania", "Willow Grove, Pennsylvania", "Hatboro, Pennsylvania", "Philadelphia, Pennsylvania", "Warminster Heights, Pennsylvania", "Oak Valley, New Jersey", "Ivyland, Pennsylvania", "Rockledge, Pennsylvania", "Cheltenham Village, Pennsylvania", "Virginia Beach, Virginia", "Camden, New Jersey", "Woodlynne, New Jersey", "Mount Ephraim, New Jersey", "Oaklyn, New Jersey", "Audubon, New Jersey", "Merchantville, New Jersey", "Westmont, New Jersey", "Glendora, New Jersey", "Trevose, Pennsylvania", "Ellisburg, New Jersey", "Hi-Nella, New Jersey", "Kingston Estates, New Jersey", "Lindenwold, New Jersey", "Penndel, Pennsylvania", "The College of New Jersey, New Jersey", "Morrisville, Pennsylvania", "Trenton, New Jersey", "Dover, New Jersey", "Somerville, New Jersey", "Victory Gardens, New Jersey", "Cape Coral, Florida", "Middletown, New York", "Bound Brook, New Jersey", "South Bound Brook, New Jersey", "Princeton Meadows, New Jersey", "Morristown, New Jersey", "Dunellen, New Jersey", "East Franklin, New Jersey", "Rutgers University-Busch Campus, New Jersey", "North Plainfield, New Jersey", "Lake Hiawatha, New Jersey", "Twin Rivers, New Jersey", "New Brunswick, New Jersey", "Palm Bay, Florida", "Plainfield, New Jersey", "Highland Park, New Jersey", "Pine Manor, Florida", "Fanwood, New Jersey", "Jamesburg, New Jersey", "Metuchen, New Jersey", "South River, New Jersey", "Garwood, New Jersey", "Caldwell, New Jersey", "Menlo Park Terrace, New Jersey", "Iselin, New Jersey", "Vauxhall, New Jersey", "Fords, New Jersey", "Singac, New Jersey", "Kiryas Joel, New York", "Connecticut Farms, New Jersey", "Union, New Jersey", "Rahway, New Jersey", "Hopelawn, New Jersey", "Roselle Park, New Jersey", "Roselle, New Jersey", "South Amboy, New Jersey", "William Paterson University of New Jersey, New Jersey", "Perth Amboy, New Jersey", "Haledon, New Jersey", "Upper Montclair, New Jersey", "East Orange, New Jersey", "Glen Ridge, New Jersey", "Watsessing, New Jersey", "Prospect Park, New Jersey", "Suffern, New York", "Carteret, New Jersey", "Ampere North, New Jersey", "Brookdale, New Jersey", "Hawthorne, New Jersey", "Lehigh Acres, Florida", "Paterson, New Jersey", "Silver Lake, New Jersey", "Schenectady, New York", "Clifton, New Jersey", "Elizabeth, New Jersey", "Newark, New Jersey", "East Newark, New Jersey", "Harrison, New Jersey", "Fair Lawn, New Jersey", "Elmwood Park, New Jersey", "Passaic, New Jersey", "Keyport, New Jersey", "North Arlington, New Jersey", "Kaser, New York", "Garfield, New Jersey", "Monsey, New York", "Wallington, New Jersey", "Staten Island, New York", "Rutherford, New Jersey", "Spring Valley, New York", "Bonita Springs, Florida", "Wood-Ridge, New Jersey", "Lodi, New Jersey", "Mount Ivy, New York", "Bayonne, New Jersey", "Hillcrest, New York", "New Square, New York", "Hasbrouck Heights, New Jersey", "Maywood, New Jersey", "Keansburg, New Jersey", "Hackensack, New Jersey", "River Edge, New Jersey", "Poughkeepsie, New York", "West Haverstraw, New York", "Jersey City, New Jersey", "North Middletown, New Jersey", "Little Ferry, New Jersey", "New Milford, New Jersey", "Bogota, New Jersey", "Albany, New York", "Ridgefield Park, New Jersey", "Union City, New Jersey", "Bergenfield, New Jersey", "Hoboken, New Jersey", "Dumont, New Jersey", "West New York, New Jersey", "Fairview, New Jersey", "Palisades Park, New Jersey", "Guttenberg, New Jersey", "Peekskill, New York", "Leonia, New Jersey", "Cliffside Park, New Jersey", "Englewood, New Jersey", "Red Bank, New Jersey", "Fort Lee, New Jersey", "Edgewater, New Jersey", "Nyack, New York", "Manhattan, New York", "Watervliet, New York", "Mechanicville, New York", "Point Pleasant, New Jersey", "Brooklyn, New York", "Highlands, New Jersey", "New York, New York", "Belmar, New Jersey", "Lake Como, New Jersey", "Asbury Park, New Jersey", "Bradley Beach, New Jersey", "Long Branch, New Jersey", "Ocean Grove, New Jersey", "Yonkers, New York", "Loch Arbour, New Jersey", "Golden Gate, Florida", "Bronx, New York", "Bronxville, New York", "Tuckahoe, New York", "Mount Vernon, New York", "Naples Manor, Florida", "New Rochelle, New York", "White Plains, New York", "Queens, New York", "Larchmont, New York", "Saddle Rock Estates, New York", "Great Neck, New York", "Great Neck Plaza, New York", "University Gardens, New York", "Great Neck Gardens, New York", "Russell Gardens, New York", "Kensington, New York", "Manorhaven, New York", "Inwood, New York", "Thomaston, New York", "Bellerose Terrace, New York", "Port Washington North, New York", "Port Chester, New York", "Bellerose, New York", "Baxter Estates, New York", "Cedarhurst, New York", "South Valley Stream, New York", "Woodmere, New York", "Elmont, New York", "Floral Park, New York", "Byram, Connecticut", "North Valley Stream, New York", "South Floral Park, New York", "Munsey Park, New York", "Valley Stream, New York", "North New Hyde Park, New York", "Manhasset Hills, New York", "New Hyde Park, New York", "Stewart Manor, New York", "Hewlett, New York", "Franklin Square, New York", "Herricks, New York", "Garden City Park, New York", "North Lynbrook, New York", "Malverne, New York", "Lynbrook, New York", "Garden City South, New York", "Albertson, New York", "Malverne Park Oaks, New York", "Williston Park, New York", "East Rockaway, New York", "West Hempstead, New York", "Mineola, New York", "Harbor Isle, New York", "Lakeview, New York", "Long Beach, New York", "Island Park, New York", "Rockville Centre, New York", "Stamford, Connecticut", "Oceanside, New York", "Carle Place, New York", "South Hempstead, New York", "Port St. Lucie, Florida", "Westbury, New York", "Baldwin, New York", "Uniondale, New York", "Danbury, Connecticut", "Burlington, Vermont", "Roosevelt, New York", "New Cassel, New York", "Freeport, New York", "Salisbury, New York", "East Meadow, New York", "North Merrick, New York", "Point Lookout, New York", "Merrick, New York", "North Bellmore, New York", "Hicksville, New York", "Winooski, Vermont", "Bellmore, New York", "Levittown, New York", "North Wantagh, New York", "Seaford, New York", "Plainedge, New York", "North Massapequa, New York", "Massapequa, New York", "Farmingdale, New York", "South Farmingdale, New York", "Massapequa Park, New York", "Huntington Station, New York", "East Massapequa, New York", "North Amityville, New York", "North Lindenhurst, New York", "Copiague, New York", "Wyandanch, New York", "Lindenhurst, New York", "North Babylon, New York", "Bridgeport, Connecticut", "North Bay Shore, New York", "Brentwood, New York", "Bay Shore, New York", "Waterbury, Connecticut", "Stony Brook University, New York", "New Haven, Connecticut", "Patchogue, New York", "Cabana Colony, Florida", "New Britain, Connecticut", "Montpelier, Vermont", "West Palm Beach, Florida", "Lake Belvedere Estates, Florida", "Stacey Street, Florida", "Royal Palm Estates, Florida", "Palm Beach Shores, Florida", "Hartford, Connecticut", "Westgate, Florida", "Greenacres, Florida", "Pine Air, Florida", "Acacia Villas, Florida", "Palm Springs, Florida", "Springfield, Massachusetts", "San Castle, Florida", "South Palm Beach, Florida", "Watergate, Florida", "Briny Breezes, Florida", "Coral Springs, Florida", "Hillsboro Pines, Florida", "Tamarac, Florida", "Sunrise, Florida", "Highland Beach, Florida", "Margate, Florida", "North Lauderdale, Florida", "Deerfield Beach, Florida", "Lauderhill, Florida", "Davie, Florida", "Lauderdale Lakes, Florida", "Pompano Beach, Florida", "Pembroke Pines, Florida", "Stock Island, Florida", "Oakland Park, Florida", "Miramar, Florida", "Roosevelt Gardens, Florida", "Broadview Park, Florida", "Lazy Lake, Florida", "Wilton Manors, Florida", "Sea Ranch Lakes, Florida", "Lauderdale-by-the-Sea, Florida", "Fort Lauderdale, Florida", "Palm Springs North, Florida", "Country Club, Florida", "Hialeah Gardens, Florida", "Miami Lakes, Florida", "Hollywood, Florida", "Miami Gardens, Florida", "Doral, Florida", "Hialeah, Florida", "West Park, Florida", "Tamiami, Florida", "Kendall West, Florida", "Ives Estates, Florida", "The Hammocks, Florida", "Kendale Lakes, Florida", "Fountainebleau, Florida", "Hallandale Beach, Florida", "Golden Glades, Florida", "Virginia Gardens, Florida", "Ojus, Florida", "Westchester, Florida", "Westwood Lakes, Florida", "West Little River, Florida", "North Miami Beach, Florida", "Aventura, Florida", "Pinewood, Florida", "The Crossings, Florida", "Country Walk, Florida", "Golden Beach, Florida", "Gladeview, Florida", "North Miami, Florida", "Brownsville, Florida", "West Miami, Florida", "Coral Terrace, Florida", "Sunny Isles Beach, Florida", "Richmond West, Florida", "Norwich, Connecticut", "Richmond Heights, Florida", "Bay Harbor Islands, Florida", "Bal Harbour, Florida", "South Miami, Florida", "North Bay Village, Florida", "Surfside, Florida", "Miami, Florida", "Palmetto Estates, Florida", "South Miami Heights, Florida", "West Perrine, Florida", "Miami Beach, Florida", "Naranja, Florida", "Leisure City, Florida", "Homestead, Florida", "Key Biscayne, Florida", "Worcester, Massachusetts", "Leominster, Massachusetts", "Concord, New Hampshire", "Nashua, New Hampshire", "Manchester, New Hampshire", "Woonsocket, Rhode Island", "Lowell, Massachusetts", "Providence, Rhode Island", "Central Falls, Rhode Island", "Pawtucket, Rhode Island", "Lawrence, Massachusetts", "Watertown Town, Massachusetts", "Cambridge, Massachusetts", "Medford, Massachusetts", "Somerville, Massachusetts", "Boston, Massachusetts", "Melrose, Massachusetts", "Malden, Massachusetts", "Everett, Massachusetts", "Chelsea, Massachusetts", "Revere, Massachusetts", "Quincy, Massachusetts", "Brockton, Massachusetts", "Lynn, Massachusetts", "Salem, Massachusetts", "New Bedford, Massachusetts", "Portland, Maine", "Juneau, Alaska", "Anchorage, Alaska", "Aguadilla, Puerto Rico", "Caban, Puerto Rico", "Aguada, Puerto Rico", "Anasco, Puerto Rico", "Puerto Real, Puerto Rico", "Rafael Gonzalez, Puerto Rico", "Arecibo, Puerto Rico", "Imbery, Puerto Rico", "Yauco, Puerto Rico", "Manati, Puerto Rico", "Adjuntas, Puerto Rico", "Jayuya, Puerto Rico", "Penuelas, Puerto Rico", "Villas del Sol, Puerto Rico", "Villa Calma, Puerto Rico", "Ponce, Puerto Rico", "Campanilla, Puerto Rico", "Las Gaviotas, Puerto Rico", "Villa de Sabana, Puerto Rico", "Luis Llorens Torres, Puerto Rico", "Sabana Seca, Puerto Rico", "Juana Diaz, Puerto Rico", "Pajaros, Puerto Rico", "Naranjito, Puerto Rico", "Bayamon, Puerto Rico", "Potala Pastillo, Puerto Rico", "Guaynabo, Puerto Rico", "Cano Martin Pena, Puerto Rico", "Coamo, Puerto Rico", "Comerio, Puerto Rico", "San Juan, Puerto Rico", "Aibonito, Puerto Rico", "Carolina, Puerto Rico", "Trujillo Alto, Puerto Rico", "Cidra, Puerto Rico", "Hacienda San Jose, Puerto Rico", "Santa Barbara, Puerto Rico", "Salinas, Puerto Rico", "Loiza, Puerto Rico", "Cayey, Puerto Rico", "Valle Hill, Puerto Rico", "Villa Hugo I, Puerto Rico", "Caguas, Puerto Rico", "Suarez, Puerto Rico", "San Isidro, Puerto Rico", "Parcelas de Navarro, Puerto Rico", "Gurabo, Puerto Rico", "Campo Rico, Puerto Rico", "Rio Grande, Puerto Rico", "Juncos, Puerto Rico", "Luquillo, Puerto Rico", "Arroyo, Puerto Rico", "Naguabo, Puerto Rico", "Kailua, Hawaii", "Kaneohe, Hawaii", "Honolulu, Hawaii", "Helemano, Hawaii", "Aiea, Hawaii", "Halawa, Hawaii", "Mililani Mauka, Hawaii", "Waimalu, Hawaii", "Wahiawa, Hawaii", "Waipio Acres, Hawaii", "Mililani Town, Hawaii", "Waipio, Hawaii", "Schofield Barracks, Hawaii", "Waikele, Hawaii", "Waipahu, Hawaii", "Iroquois Point, Hawaii", "West Loch Estate, Hawaii", "Ewa Beach, Hawaii", "Ewa Gentry, Hawaii", "Ewa Villages, Hawaii", "Ocean Pointe, Hawaii", "Makakilo, Hawaii", "Kapolei, Hawaii", "Maili, Hawaii"]
	super_likes := 0
	first_loop := true
	; Potentially infiite loading
	neutral_profile := false
	
	navigate_to_discover(dating_app)
	
	; like test
	; like(dating_app, root_directory)
	; super_like(dating_app, root_directory)
	; exitApp
	
	super_likes := remaining_super_likes(dating_app)
	
	if super_likes == "o" or super_likes == "O"
	{
		super_likes := 0
	}
	
	if dating_app == "photofeeler"
	{
		super_likes := 2000000000
	}
	
	sleep 500
	
	while true
	{	
		start:
		
		; Refresh Page Logics
		if dating_app == "tinder"
		{			
			; Click refresh
			mouseClick "left", 100, 67			 
			sleep 15000
			
			; Bad Gateway
			if InStr(ocr(900, 135, 964, 173, 100), "Bad", 0)
			{
				goto("start")
			}
			
			; Click away "____ likes you"
			mouseClick "left", 1300, 242
			sleep 500
			
			; Reset image to position 1
			mouseClick "left", 989, 519
			sleep 500
		}
		else if dating_app == "bumble"
		{
			if neutral_profile
			{		
				targeted_cities_index += 1
				restart_bumble()
				neutral_profile := false
				goto("start")
			}
			
			; Restart from beginning if at the end
			if targeted_cities_index > 1448
			{
				targeted_cities_index := 0
			}
		
		
			; Click away any popups including "Second time's a charm" compliment suggestion and "It's a match!"
			mouseClick "left", 701, 71
			sleep 1000	
			; Click discover to close side bar if above popup didn't happen
			MouseClick "left", 962, 1042	
			sleep 1500
			
			; To detect when you run out of people ("Adjust your filters")
			; if true  ; test
			if first_loop or InStr(ocr(831, 783, 932, 828, 100), "Adjust", 0)
			{
				first_loop := false			
			
				; Click profile
				MouseClick "left", 718, 1041
				sleep 1000
				
				; Click travel mode
				MouseClick "left", 893, 295
				sleep 2000
				
				targeted_cities_index += 1
				send targeted_cities[targeted_cities_index]				
				sleep 5000	
				; Fix for unable to click
				send "{enter}"
				sleep 500
				
				; Click the first city in the list
				MouseClick "left", 957, 380		
				sleep 4000		
				
				; Click confirmation popup
				MouseClick "left", 960, 980		
				sleep 2000
				
				; Reset, restart Bumble and redo search if it stalls
				if InStr(ocr(1145, 85, 1253, 127, 500), "Cancel", 0)
				{
					; Location is in database
					if !InStr(ocr(938, 941, 1068, 973, 500), "database", 0)
					{
						; Go back one so when it runs again, it will add one and redo the search
						targeted_cities_index -= 1
						
					}
					
					first_loop := true
					
					restart_bumble()
				}
					
				
				goto("start")
			}
		}
		else if dating_app == "hinge"
		{
			; search for "skipped", then refreshes
			if InStr(ocr(910, 775, 1000, 808, 500), "sk", 0)
			{
				; Go into match preferences
				mouseClick "left", 943, 688
				sleep 5000
				; Go back
				mouseClick "left", 1229, 69
				sleep 7000
				goto("start")
			}
			
			; search for "skipped" text or "Try" in "Try Again", then refresh matches
			if InStr(ocr(900, 656, 946, 690, 500), "Try", 0)
			{
				; Click "Try Again"
				mouseClick "left", 957, 670
				sleep 500
				mouseClick "left", 957, 670
				sleep 7000
				goto("start")
			}
		}
		else if dating_app == "photofeeler" 
		{			
			; Test open_ai_clip
			; if InStr(ocr(387, 338, 424, 361, 100), "Aw", 0)
			
			if InStr(ocr(769, 145, 925, 245, 100), "Max", 0) or InStr(ocr(387, 338, 424, 361, 100), "Aw", 0)
			{
				; F5 for refresh stops working after repeated uses
				mouseClick "left", 103, 69
				
				sleep 100
				A_Clipboard := ""
				sleep 100
				 
				sleep 10000
				
				goto("start")
			}	
			
			; No more pictures to vote
			if InStr(ocr(660, 205, 772, 251, 100), "Credits", 0)
			{
				send "{LControl down}l"
				send "{LControl up}"
				send "https://www.photofeeler.com/vote/dating"
				send "{Enter}"
				sleep 3000
			}
		}
		
		take_screenshot(dating_app)  
		
		returned_list := make_decision()
		decision := returned_list[1]
		adjective := returned_list[2]
		
		if adjective == "neutral"
		{
			neutral_profile := true
		}
		else
		{
			neutral_profile := false
		}
		
		; Liking Logic
		if decision == "super_like" && super_likes > 0
		{
			super_like(dating_app, root_directory)
			super_likes -= 1
		}
		else if decision == "super_like" || decision == "like"
		{
			hinge_opener := like(dating_app, root_directory)
		}
		else
		{
			dislike(dating_app)
		}		
		
		if dating_app == "tinder"
		{
			; sleep 6000
		}
		else if dating_app == "bumble"
		{
		}
		else if dating_app == "hinge"
		{
			sleep 3000
			; Below cause errors when notification of someone matching with you occurs at the same time as we are clicking off prompt poll popup, it will click the match popup and go to messages page.
			; Click off add a prompt poll popup
			; mouseClick "left", 723, 161
			; sleep 3000
			; mouseClick "left", 828, 567
			; sleep 3000			
			; If no prompt poll, need to remove photo description as a consequence of clicking on it
			; mouseClick "left", 957, 634
		}
		else if dating_app == "photofeeler"
		{
		}
		
		clear_screenshot_directory(root_directory, dating_app)
		
		;if dating_app == "tinder"
		;{
		;	sleep 5000
		;}
	}
	
}

; "tinder", "bumble", "okcupid", "match", "eharmony", "hinge"

; "1366x768"
; main("photofeeler", "C:\Users\LENOVO\Desktop\GitHub\Operation_Love")

; "1920x1080"
main("tinder", "C:\Users\Bull\Desktop\Github\Operation_Love")

; "1366x768"
; main("bumble", "C:\Users\Jack.Wu\Documents\GitHub\Operation_Love")

; "1920x1080"
; main("hinge", "C:\Users\Jack.Wu\Documents\GitHub\Operation_Love")

f12::
{
	global targeted_cities_index
	
	; May over-subtract, but prevents missing a city
	targeted_cities_index -= 1
	
	FileDelete "targeted_cities_index.txt"
	FileAppend targeted_cities_index, "targeted_cities_index.txt"
	
	exitApp
}